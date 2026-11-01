import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { GeometryDocument, GeometryObject } from './geometryModel';
import './GeometryViewport.css';

export type GeometryViewportProps = {
  document: GeometryDocument;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  /** Change when opening a different document to fit it without disrupting edits. */
  resetKey?: string | number;
};
type CameraView = 'iso' | 'xy' | 'xz' | 'yz';
type ViewportEngine = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  model: THREE.Group;
  guides: THREE.Group;
  meshes: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>[];
  grid: THREE.GridHelper | null;
  domain: THREE.LineSegments | null;
  fit: (view?: CameraView) => void;
  render: () => void;
  distance: number;
};

function surfaceColors(index: number, selected: boolean) {
  return selected
    ? { surface: 0xf2b969, emissive: 0x39230d, edge: 0xffdeac }
    : index <= 1.6
      ? { surface: 0x344d64, emissive: 0x07121d, edge: 0x718b9e }
      : { surface: 0x4ad4bf, emissive: 0x062523, edge: 0x93f4e2 };
}

function disposeGroup(group: THREE.Group) {
  group.traverse(node => {
    const drawable = node as THREE.Mesh;
    drawable.geometry?.dispose();
    if (drawable.material) {
      const materials = Array.isArray(drawable.material) ? drawable.material : [drawable.material];
      materials.forEach(material => {
        const mapped = material as THREE.MeshBasicMaterial;
        mapped.map?.dispose();
        material.dispose();
      });
    }
  });
  group.clear();
}

function makeGeometry(object: GeometryObject): THREE.BufferGeometry {
  if (object.kind === 'box') return new THREE.BoxGeometry(object.size.x, object.size.y, object.size.z);
  if (object.kind === 'cylinder') {
    const cylinder = new THREE.CylinderGeometry(object.radius, object.radius, object.size.z, 64);
    cylinder.rotateX(Math.PI / 2);
    return cylinder;
  }
  // Extruding an annulus gives a rectangular waveguide cross-section, not a torus.
  const outline = new THREE.Shape();
  outline.absarc(0, 0, object.radius, 0, Math.PI * 2, false);
  const hole = new THREE.Path();
  hole.absarc(0, 0, object.inner_radius, 0, Math.PI * 2, true);
  outline.holes.push(hole);
  const ring = new THREE.ExtrudeGeometry(outline, { depth: object.size.z, bevelEnabled: false, curveSegments: 64, steps: 1 });
  ring.translate(0, 0, -object.size.z / 2);
  return ring;
}

function axisLabel(text: string, color: string): THREE.Sprite {
  const canvas = window.document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context) {
    context.fillStyle = color; context.font = 'bold 40px sans-serif';
    context.textAlign = 'center'; context.textBaseline = 'middle';
    context.fillText(text, 32, 33);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
}

export function GeometryViewport({ document, selectedId, onSelect, resetKey }: GeometryViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<ViewportEngine | null>(null);
  const documentRef = useRef(document);
  const selectionRef = useRef(onSelect);
  const selectedRef = useRef(selectedId);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [gridVisible, setGridVisible] = useState(true);
  const [domainVisible, setDomainVisible] = useState(true);
  const [cameraView, setCameraView] = useState<CameraView | 'free'>('iso');
  const [distance, setDistance] = useState('');
  documentRef.current = document;
  selectionRef.current = onSelect;
  selectedRef.current = selectedId;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    setError('');
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'default' });
    } catch {
      setError('The 3D viewport needs WebGL 2. Enable hardware acceleration in your browser or open this project in another browser. Your geometry remains available in the editor.');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0b141f);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const canvas = renderer.domElement;
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Interactive 3D geometry. Drag to orbit, scroll to zoom, right-drag or arrow keys to pan. Press F to fit.');
    host.appendChild(canvas);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 10000);
    camera.up.set(0, 0, 1);
    camera.position.set(10, -13, 10);
    const controls = new OrbitControls(camera, canvas);
    // Render only on interaction or document changes; an idle editor consumes no animation loop.
    controls.enableDamping = false;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.01;
    controls.maxDistance = 10000;
    controls.listenToKeyEvents(canvas);
    scene.add(new THREE.AmbientLight(0xbacfe6, 1.9));
    const light = new THREE.DirectionalLight(0xffffff, 3.0);
    light.position.set(4, -6, 12); scene.add(light);
    const rim = new THREE.DirectionalLight(0x56b6d5, 1.6);
    rim.position.set(-8, 3, 4); scene.add(rim);
    const model = new THREE.Group(); scene.add(model);
    const guides = new THREE.Group(); scene.add(guides);
    let disposed = false;
    let contextLost = false;
    let presetChanging = false;
    let resizeObserver: ResizeObserver | undefined;
    const render = () => {
      if (disposed || contextLost) return;
      try {
        renderer.render(scene, camera);
        canvas.dataset.camera = [...camera.position.toArray(), ...controls.target.toArray()].map(v => v.toFixed(4)).join(',');
        canvas.dataset.visibleObjects = String(model.children.filter(node => node.visible).length);
        setDistance(`${camera.position.distanceTo(controls.target).toFixed(2)} µm`);
      } catch {
        setError('The 3D view could not render. Your geometry is still saved in the editor. Try reopening the viewport.');
      }
    };
    const engine: ViewportEngine = {
      renderer, scene, camera, controls, model, guides, meshes: [], grid: null, domain: null, distance: 10, render,
      fit: (view) => {
        const domain = documentRef.current.domain;
        const radius = new THREE.Vector3(domain.x, domain.y, domain.z).length() / 2;
        const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
        const horizontalFov = Math.atan(Math.tan(halfFov) * camera.aspect);
        const distance = radius / Math.sin(Math.min(halfFov, horizontalFov)) * 1.12;
        const direction = view === 'xy' ? new THREE.Vector3(0, -0.0001, 1)
          : view === 'xz' ? new THREE.Vector3(0, -1, 0)
          : view === 'yz' ? new THREE.Vector3(1, 0, 0)
          : view === 'iso' ? new THREE.Vector3(1, -1.35, 1)
          : camera.position.clone().sub(controls.target).normalize();
        engine.distance = distance;
        controls.minDistance = Math.max(radius / 200, .001);
        controls.maxDistance = Math.max(distance * 20, 10);
        camera.near = Math.max(radius / 10000, .00001);
        camera.far = Math.max(distance * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        camera.position.copy(direction.normalize().multiplyScalar(distance));
        presetChanging = true;
        controls.update();
        presetChanging = false;
        if (view) setCameraView(view);
        render();
      },
    };
    engineRef.current = engine;
    const changed = () => { if (!presetChanging) setCameraView('free'); render(); };
    controls.addEventListener('change', changed);
    const resize = () => {
      const rect = host.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
      const oldAspect = camera.aspect;
      camera.aspect = width / height; camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      // Preserve inspection framing on resize; fitting is always available in the toolbar.
      if (oldAspect === 1) engine.fit('iso'); else render();
    };
    resizeObserver = new ResizeObserver(resize); resizeObserver.observe(host); resize();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let down: { x: number; y: number; id: number } | null = null;
    const pointerDown = (event: PointerEvent) => {
      canvas.focus({ preventScroll: true });
      if (event.button === 0) down = { x: event.clientX, y: event.clientY, id: event.pointerId };
    };
    const pointerUp = (event: PointerEvent) => {
      const start = down; down = null;
      if (!start || start.id !== event.pointerId || event.button !== 0 || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 5) return;
      const rect = canvas.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(engine.meshes.filter(mesh => mesh.visible), false)[0];
      selectionRef.current(hit ? String(hit.object.userData.objectId) : null);
    };
    const pointerCancel = () => { down = null; };
    const keyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === 'f' && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); engine.fit(); }
    };
    const lost = (event: Event) => {
      event.preventDefault(); contextLost = true;
      setError('The graphics context was lost. Your geometry is unchanged. Reopen the viewport to restore the 3D view.');
    };
    canvas.addEventListener('pointerdown', pointerDown);
    canvas.addEventListener('pointerup', pointerUp);
    canvas.addEventListener('pointercancel', pointerCancel);
    canvas.addEventListener('keydown', keyDown);
    canvas.addEventListener('webglcontextlost', lost);
    engine.fit('iso');
    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      canvas.removeEventListener('pointerdown', pointerDown);
      canvas.removeEventListener('pointerup', pointerUp);
      canvas.removeEventListener('pointercancel', pointerCancel);
      canvas.removeEventListener('keydown', keyDown);
      canvas.removeEventListener('webglcontextlost', lost);
      controls.removeEventListener('change', changed); controls.dispose();
      disposeGroup(model); disposeGroup(guides);
      renderer.dispose(); renderer.forceContextLoss();
      canvas.remove();
      if (engineRef.current === engine) engineRef.current = null;
    };
  }, [attempt]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    disposeGroup(engine.model); disposeGroup(engine.guides); engine.meshes = [];
    for (const object of document.objects) {
      const selected = object.id === selectedRef.current;
      const colors = surfaceColors(object.refractive_index, selected);
      const material = new THREE.MeshStandardMaterial({ color: colors.surface, roughness: object.refractive_index <= 1.6 ? .6 : .32, metalness: .12, emissive: colors.emissive, emissiveIntensity: .24 });
      const geometry = makeGeometry(object);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(object.position.x, object.position.y, object.position.z);
      mesh.rotation.z = THREE.MathUtils.degToRad(object.rotation_deg);
      mesh.name = object.name; mesh.visible = object.visible;
      mesh.userData.objectId = object.id;
      mesh.userData.refractiveIndex = object.refractive_index;
      const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 25), new THREE.LineBasicMaterial({ color: colors.edge, transparent: true, opacity: .7 }));
      mesh.add(outline); engine.model.add(mesh); engine.meshes.push(mesh);
    }
    const domainGeometry = new THREE.BoxGeometry(document.domain.x, document.domain.y, document.domain.z);
    const border = new THREE.LineSegments(new THREE.EdgesGeometry(domainGeometry), new THREE.LineDashedMaterial({ color: 0x647d96, transparent: true, opacity: .65, dashSize: .12, gapSize: .08 }));
    domainGeometry.dispose(); border.computeLineDistances(); border.visible = domainVisible;
    engine.guides.add(border); engine.domain = border;
    const span = Math.max(document.domain.x, document.domain.y);
    const grid = new THREE.GridHelper(span, 20, 0x3b5268, 0x263748);
    grid.rotation.x = Math.PI / 2; grid.position.z = -document.domain.z / 2;
    grid.visible = gridVisible; engine.guides.add(grid); engine.grid = grid;
    const axisLength = Math.max(span * .12, .25);
    const axes = new THREE.AxesHelper(axisLength);
    axes.position.set(-document.domain.x / 2, -document.domain.y / 2, -document.domain.z / 2);
    engine.guides.add(axes);
    ([['x', '#f58c86', [axisLength * 1.2, 0, 0]], ['y', '#9bdf98', [0, axisLength * 1.2, 0]], ['z', '#85b6ff', [0, 0, axisLength * 1.2]]] as const).forEach(([label, color, offset]) => {
      const sprite = axisLabel(label, color);
      sprite.position.copy(axes.position).add(new THREE.Vector3(...offset));
      sprite.scale.setScalar(axisLength * .3); engine.guides.add(sprite);
    });
    engine.render();
  }, [document, attempt]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.meshes.forEach(mesh => {
      const selected = mesh.userData.objectId === selectedId;
      const colors = surfaceColors(Number(mesh.userData.refractiveIndex), selected);
      mesh.material.color.setHex(colors.surface);
      mesh.material.emissive.setHex(colors.emissive);
      const outline = mesh.children[0] as THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
      outline.material.color.setHex(colors.edge);
    });
    engine.render();
  }, [selectedId, attempt]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    if (engine.grid) engine.grid.visible = gridVisible;
    if (engine.domain) engine.domain.visible = domainVisible;
    engine.render();
  }, [gridVisible, domainVisible, attempt]);

  useEffect(() => { engineRef.current?.fit('iso'); }, [resetKey, attempt]);

  const zoom = (factor: number) => {
    const engine = engineRef.current; if (!engine) return;
    const delta = engine.camera.position.clone().sub(engine.controls.target);
    const length = THREE.MathUtils.clamp(delta.length() * factor, engine.controls.minDistance, engine.controls.maxDistance);
    engine.camera.position.copy(engine.controls.target).add(delta.setLength(length));
    engine.controls.update(); engine.render();
  };
  const selected = document.objects.find(object => object.id === selectedId);
  return <section className="geometry-3d" aria-label="3D geometry viewport">
    <div className="geometry-3d-toolbar">
      <div className="geometry-camera-buttons" role="group" aria-label="Camera views">
        <button type="button" onClick={() => engineRef.current?.fit()} title="Fit the whole domain (F)">Fit view</button>
        {([['iso', 'Iso', 'Isometric view'], ['xy', 'Top', 'Top view (XY)'], ['xz', 'Front', 'Front view (XZ)'], ['yz', 'Right', 'Right view (YZ)']] as const).map(([value, label, title]) => <button key={value} type="button" title={title} aria-label={title} aria-pressed={cameraView === value} onClick={() => engineRef.current?.fit(value)}>{label}</button>)}
        <button type="button" onClick={() => zoom(.8)} aria-label="Zoom in" title="Zoom in">+</button>
        <button type="button" onClick={() => zoom(1.25)} aria-label="Zoom out" title="Zoom out">−</button>
      </div>
      <div className="geometry-guide-buttons" role="group" aria-label="Viewport guides">
        <button type="button" aria-pressed={gridVisible} onClick={() => setGridVisible(value => !value)}>Grid</button>
        <button type="button" aria-pressed={domainVisible} onClick={() => setDomainVisible(value => !value)}>Domain</button>
      </div>
    </div>
    <div className="geometry-3d-stage">
      <div className="geometry-3d-host" ref={hostRef}/>
      <div className="geometry-3d-caption"><span className="geometry-preview-pill">{document.mode === 'slab-preview' ? 'EXTRUDED 2D PREVIEW' : '3D GEOMETRY DRAFT'}</span><span>{document.domain.x} × {document.domain.y} × {document.domain.z} µm</span></div>
      <div className="geometry-3d-axis-key" aria-hidden="true"><span>x</span><span>y</span><span>z ↑</span></div>
      {error && <div className="geometry-3d-error" role="alert"><strong>3D view unavailable</strong><p>{error}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Reopen viewport</button></div>}
      <div className="geometry-3d-selection">{selected ? <><span className="geometry-selection-dot"/>{selected.name}{!selected.visible && ' · hidden'}<span>n = {selected.refractive_index}</span></> : 'Click a solid to select it'}</div>
    </div>
    <div className="geometry-3d-help"><span>Drag to orbit · scroll to zoom · right-drag to pan</span><span title="Distance from camera to orbit center">{distance || 'Perspective'} · {document.objects.filter(object => object.visible).length} visible</span></div>
  </section>;
}

export default GeometryViewport;
