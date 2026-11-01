# Optional browser verification

The application build uses the pinned dependencies in `frontend/pnpm-lock.yaml`.
Playwright is a separate QA tool and is not required to build or run the app.

The browser journeys were tested with the bundled **Playwright 1.62.1**, **Node
24.22.1**, and installed **Google Chrome on Windows**, against the local WSL API.
Meep CPU, FDTDX CPU, and FDTDX GPU each completed an edited-width simulation,
displayed actual fields and the material grid, and downloaded the numerical ZIP.
The suite also checks explicit device selection, invalid physical inputs, and
immutable run configurations. The final Meep/GPU checks additionally cover
partial JSON imports and layouts at 1440, 1280, and 390 px without horizontal
overflow. Linux-hosted browser execution has not been separately verified.

To install optional QA dependencies separately, run these commands from the
repository root with Node/npm available:

```bash
npm install --prefix .local/browser-qa --no-save --package-lock=false playwright@1.62.1
```

Start the real application on `http://127.0.0.1:8000`. With Google Chrome already
installed on the machine running the test:

```bash
NODE_PATH="$PWD/.local/browser-qa/node_modules" \
PHOTONICS_BROWSER_CHANNEL=chrome \
PHOTONICS_TEST_TARGET=meep:cpu \
PHOTONICS_ARTIFACT_DIR=.local/browser-tests/meep \
node frontend/browser-smoke.cjs
```

Repeat with `PHOTONICS_TEST_TARGET=fdtdx:cpu` or `fdtdx:gpu` only when that target
is actually available. A GPU request must fail if the GPU is unavailable. The
tests execute real simulations; run one test at a time.

For a separate Linux Chromium setup, install the browser and its OS dependencies
using Playwright's CLI, then omit `PHOTONICS_BROWSER_CHANNEL`:

```bash
node .local/browser-qa/node_modules/playwright/cli.js install --with-deps chromium
```

This separate-install recipe follows the [official Playwright browser setup
documentation](https://playwright.dev/docs/browsers); the event verification used
the existing bundled library and Chrome, not a fresh browser installation.
The test writes original application screenshots, a downloaded ZIP, and a JSON
report to the selected artifact directory. No solver responses are mocked.

## Time evolution and run passport

The v0.2 upgrade test uses the same separate Playwright setup and a running local
API. It starts real simulations and checks recorded-frame pixels against the
actual arrays, a fixed amplitude scale, exact simulated times, play/pause, the
unchanged final snapshot, passport fingerprints, and configuration provenance.

```bash
NODE_PATH="$PWD/.local/browser-qa/node_modules" \
PHOTONICS_BROWSER_CHANNEL=chrome \
PHOTONICS_TEST_TARGET=meep:cpu \
PHOTONICS_ARTIFACT_DIR=.local/browser-upgrades \
node frontend/browser-upgrade-smoke.cjs
```

The negative test deliberately aborts one frame-fetch request to verify a
visible error and a successful retry. This is transport-failure injection only:
numerical fields are never fabricated or replaced. The test also starts a
second run to ensure playback does not retain the first run's frames.
It captures two original screenshots and a JSON report. The full upgraded
browser journey was verified with Meep CPU; frame/passport HTTP checks passed
for Meep CPU, FDTDX CPU, and FDTDX GPU. See
[the browser upgrade report](verification/browser-upgrades.json) and
[the backend evidence](verification/run-passport.json).

## Guided-port measurements

The v0.5 port test starts a real reference/device pair, checks each plotted
sample and its signed power normalization, downloads CSV/JSON/ZIP, inspects
the passport, and verifies explicit target selection and mobile layout:

```bash
NODE_PATH="$PWD/.local/browser-qa/node_modules" \
PHOTONICS_BROWSER_CHANNEL=chrome \
PHOTONICS_SOLVER=meep PHOTONICS_DEVICE=cpu \
PHOTONICS_ARTIFACT_DIR=.local/browser-ports \
node frontend/browser-ports-smoke.cjs
```

Use `PHOTONICS_SOLVER=fdtdx` and an explicitly available CPU/GPU device to test
those routes, one job at a time. `PHOTONICS_PORT_RUN_ID` reopens an existing
completed port run without launching another solve. The masked-bin and failed
HTTP cases are labelled UI fixtures, separate from the actual solver evidence.
The test does not invent numerical measurements.
