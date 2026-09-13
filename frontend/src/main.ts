import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './style.css';

type Neuron = {
  id: number;
  root_id?: string | null;
  x: number;
  y: number;
  z: number;
  region: string;
  super_class?: string;
  side?: string;
};

type LayoutResponse = {
  neurons: Neuron[];
  source: string;
  count: number;
  total_connectome_neurons?: number;
  coordinate_kind?: string;
};

type Telemetry = {
  type: 'telemetry';
  telemetry_source?: string;
  trial: number;
  problem: string;
  answer: number;
  correct: boolean;
  reward: number;
  accuracy: number;
  activity: [number, number][];
  plasticity: {
    mean_delta_w: number;
    active_synapses: number;
  };
};

const canvas = document.querySelector<HTMLCanvasElement>('#brain');
if (!canvas) throw new Error('Missing #brain canvas');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x090b10, 0.018);

const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 1.5, 23);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 4;
controls.maxDistance = 45;

const regionBase: Record<string, THREE.Color> = {
  optic: new THREE.Color('#6f91b8'),
  central: new THREE.Color('#84a88d'),
  sensory: new THREE.Color('#d0a65d'),
  visual_projection: new THREE.Color('#7f87c8'),
  visual_centrifugal: new THREE.Color('#927cc4'),
  descending: new THREE.Color('#c17874'),
  ascending: new THREE.Color('#68a9a2'),
  motor: new THREE.Color('#c58b62'),
  endocrine: new THREE.Color('#b178a5'),
  visual: new THREE.Color('#6a8caf'),
  mushroom_body: new THREE.Color('#9c7fc0'),
  dopamine: new THREE.Color('#c28f72'),
  mock: new THREE.Color('#8590a0'),
};

const fallbackColor = new THREE.Color('#8590a0');
const hotColor = new THREE.Color('#ffffff');

let pointCloud: THREE.Points | null = null;
let colors: Float32Array | null = null;
let neurons: Neuron[] = [];
let previousActiveIndices: number[] = [];
let layoutSummary = 'loading layout';

const byId = new Map<number, number>();
const problemEl = document.querySelector<HTMLElement>('#problem')!;
const answerEl = document.querySelector<HTMLElement>('#answer')!;
const accuracyEl = document.querySelector<HTMLElement>('#accuracy')!;
const trialEl = document.querySelector<HTMLElement>('#trial')!;
const plasticityEl = document.querySelector<HTMLElement>('#plasticity')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;

function setNeuronColor(index: number, activity: number) {
  if (!colors) return;
  const neuron = neurons[index];
  const base = regionBase[neuron.region] ?? fallbackColor;
  const mixed = base.clone().lerp(hotColor, Math.min(1, activity));
  const offset = index * 3;
  colors[offset] = mixed.r;
  colors[offset + 1] = mixed.g;
  colors[offset + 2] = mixed.b;
}

function initializeColors() {
  neurons.forEach((_, index) => setNeuronColor(index, 0));
}

function clearPreviousActivity() {
  for (const index of previousActiveIndices) setNeuronColor(index, 0);
  previousActiveIndices = [];
}

async function loadLayout() {
  const response = await fetch('http://localhost:8000/api/layout');
  if (!response.ok) throw new Error(`Layout request failed: ${response.status}`);
  const data = (await response.json()) as LayoutResponse;
  neurons = data.neurons;

  const positions = new Float32Array(neurons.length * 3);
  colors = new Float32Array(neurons.length * 3);

  neurons.forEach((neuron, index) => {
    byId.set(neuron.id, index);
    const p = index * 3;
    positions[p] = neuron.x;
    positions[p + 1] = neuron.y;
    positions[p + 2] = neuron.z;
  });

  initializeColors();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  const material = new THREE.PointsMaterial({
    size: neurons.length > 5000 ? 0.065 : 0.09,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    sizeAttenuation: true,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  const total = data.total_connectome_neurons;
  const subsetLabel = data.coordinate_kind === 'soma' && total
    ? `${data.count.toLocaleString()} real soma positions / ${total.toLocaleString()} total neurons`
    : `${data.count.toLocaleString()} neurons`;
  layoutSummary = `${subsetLabel} · ${data.source}`;
  statusEl.textContent = layoutSummary;
}

function connectTelemetry() {
  const socket = new WebSocket('ws://localhost:8000/ws/telemetry');

  socket.addEventListener('open', () => {
    statusEl.textContent = `${layoutSummary} · live 10 Hz`;
  });

  socket.addEventListener('message', (event) => {
    const frame = JSON.parse(event.data) as Telemetry;
    if (frame.type !== 'telemetry' || !pointCloud || !colors) return;

    clearPreviousActivity();
    for (const [id, activity] of frame.activity) {
      const index = byId.get(id);
      if (index !== undefined) {
        setNeuronColor(index, activity);
        previousActiveIndices.push(index);
      }
    }

    const colorAttr = pointCloud.geometry.getAttribute('color') as THREE.BufferAttribute;
    colorAttr.needsUpdate = true;

    problemEl.textContent = frame.problem;
    answerEl.textContent = `${frame.answer} ${frame.correct ? '✓' : '✕'}`;
    accuracyEl.textContent = `${(frame.accuracy * 100).toFixed(1)}%`;
    trialEl.textContent = frame.trial.toString();
    plasticityEl.textContent = `${frame.plasticity.active_synapses} · Δw ${frame.plasticity.mean_delta_w >= 0 ? '+' : ''}${frame.plasticity.mean_delta_w}`;

    if (frame.telemetry_source === 'mock') {
      statusEl.textContent = `${layoutSummary} · mock activity at 10 Hz`;
    }
  });

  socket.addEventListener('close', () => {
    statusEl.textContent = `${layoutSummary} · telemetry disconnected · retrying…`;
    window.setTimeout(connectTelemetry, 1500);
  });

  socket.addEventListener('error', () => socket.close());
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

window.addEventListener('resize', resize);

function animate() {
  controls.update();
  if (pointCloud) pointCloud.rotation.y += 0.00035;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

loadLayout()
  .then(connectTelemetry)
  .catch((error) => {
    console.error(error);
    statusEl.textContent = 'backend unavailable';
  });

animate();
