"""Open a local workbench server in a dedicated Chromium application window."""
import fcntl
import hashlib
import os
from pathlib import Path, PureWindowsPath
import shutil
import signal
import socket
import subprocess
import threading
import time


def _browser(requested=None):
    if requested:
        value = shutil.which(os.fspath(requested))
        if not value or not Path(value).is_file():
            raise ValueError('The requested browser executable could not be found.')
        return Path(value).resolve()
    for name in ('google-chrome', 'chromium', 'chromium-browser', 'microsoft-edge'):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    if os.environ.get('WSL_DISTRO_NAME'):
        for name in ('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
                     '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
                     '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe'):
            if Path(name).is_file() and os.access(name, os.X_OK):
                return Path(name)
    raise RuntimeError('Install Chrome, Chromium or Edge, or pass --browser with its executable path.')


def _profile(data, browser):
    # Keep Chromium away from the user's ordinary browser profile. On WSL,
    # host Chrome keeps its profile on a Windows drive for reliable locking.
    profile = data / '.local' / 'desktop-profile'
    if browser.suffix.lower() != '.exe':
        profile.mkdir(mode=0o700, parents=True, exist_ok=True)
        return str(profile)
    if not os.environ.get('WSL_DISTRO_NAME'):
        raise ValueError('Windows browser executables require WSL interoperation.')
    windows_path = subprocess.check_output(['wslpath', '-w', str(profile)], text=True).strip()
    if len(PureWindowsPath(windows_path).drive) == 2:
        profile.mkdir(mode=0o700, parents=True, exist_ok=True)
        return windows_path
    # Read the Windows user's LocalAppData rather than assuming a user name,
    # system drive or code page. PowerShell writes just the UTF-8 path.
    script = "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); [Environment]::GetFolderPath('LocalApplicationData')"
    base = subprocess.check_output(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
                                   encoding='utf-8', timeout=15).strip()
    if not PureWindowsPath(base).is_absolute():
        raise RuntimeError('Could not find Windows LocalAppData for the desktop browser profile.')
    identity = os.environ['WSL_DISTRO_NAME'] + '\n' + str(data)
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return str(PureWindowsPath(base, 'Photonics Workbench', 'Profiles', key))


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _close_window(window, executable, url, profile):
    if window.poll() is not None:
        return
    if executable.suffix.lower() == '.exe':
        # A WSL relay PID is different from a Windows browser PID. Send a
        # window-close request only to the browser with our exact origin and
        # profile. Values travel in the child environment, never as shell code.
        script = r"""
$origin = [Environment]::GetEnvironmentVariable('PHOTONICS_DESKTOP_ORIGIN')
$profilePath = [Environment]::GetEnvironmentVariable('PHOTONICS_DESKTOP_PROFILE')
$originFlag = '(?:^|\s)"?' + [regex]::Escape('--app=' + $origin) + '"?(?=\s|$)'
$profileFlag = '(?:^|\s)"?' + [regex]::Escape('--user-data-dir=' + $profilePath) + '"?(?=\s|$)'
Get-CimInstance Win32_Process -Filter "Name = 'chrome.exe' OR Name = 'msedge.exe' OR Name = 'chromium.exe'" |
    Where-Object { $_.CommandLine -match $originFlag -and $_.CommandLine -match $profileFlag } |
    ForEach-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if ($p) { [void]$p.CloseMainWindow() } }
"""
        env = dict(os.environ, PHOTONICS_DESKTOP_ORIGIN=url, PHOTONICS_DESKTOP_PROFILE=profile)
        env['WSLENV'] = ':'.join(filter(None, [env.get('WSLENV'), 'PHOTONICS_DESKTOP_ORIGIN', 'PHOTONICS_DESKTOP_PROFILE']))
        try:
            subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
                           env=env, check=True, timeout=10, stdout=subprocess.DEVNULL)
            window.wait(timeout=5)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    window.terminate()
    try:
        window.wait(timeout=5)
    except subprocess.TimeoutExpired:
        window.kill()
        window.wait()


def launch_desktop(data, port, browser=None):
    """Keep the owned server alive until its window closes and active runs finish."""
    import uvicorn

    if not 1 <= port <= 65535:
        raise ValueError('Choose a desktop port from 1 to 65535.')
    executable = _browser(browser)
    data = Path(data).expanduser().resolve()
    state = data / '.local'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'web.lock').open('a') as lock, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another workbench already owns this data directory. Close it or choose a separate --data-dir.') from error
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', port))
            listener.listen()
        except OSError as error:
            raise RuntimeError(f'Desktop port {port} is unavailable. Close the service using it or choose --port. Keep the same port for the same saved draft.') from error
        profile = _profile(data, executable)
        url = f'http://127.0.0.1:{port}'
        from backend.app import app, manager
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, name='photonics-desktop-http')
        window = None
        previous_signal = signal.signal(signal.SIGTERM, _interrupt)
        thread.start()
        try:
            deadline = time.monotonic() + 20
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError('The local workbench could not start. Check whether another instance owns the run store.')
                time.sleep(.05)
            print(f'Opening Photonics Workbench at {url}. Close the application window to finish.', flush=True)
            opened = time.monotonic()
            window = subprocess.Popen([str(executable), f'--app={url}', f'--user-data-dir={profile}',
                                       '--no-first-run', '--no-default-browser-check', '--disable-background-mode',
                                       '--disable-background-networking', '--disable-sync'], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while window.poll() is None:
                if not thread.is_alive():
                    raise RuntimeError('The owned local workbench server stopped.')
                time.sleep(.25)
            if window.returncode != 0 or time.monotonic() - opened < 1:
                raise RuntimeError('The application browser exited during launch. Close other windows using this dedicated profile and try again.')
            while True:
                with manager.lock:
                    active = any(job['status'] in ('queued', 'running') for job in manager.jobs.values())
                if not active:
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            print('Closing the workbench window and its local server.', flush=True)
            return 130
        finally:
            try:
                if window is not None:
                    _close_window(window, executable, url, profile)
            finally:
                server.should_exit = True
                thread.join()
                signal.signal(signal.SIGTERM, previous_signal)
    return 0
