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
};

type CoreMetrics = {
  overall: number | null;
  balanced_accuracy?: number | null;
  mean_absolute_error?: number | null;
  within_one_accuracy?: number | null;
  recent_20: number | null;
  recent_100: number | null;
  recent_500: number | null;
  successes: number;
  attempts: number;
  by_target_accuracy?: Record<string, number | null>;
};

type SuccessMetrics = CoreMetrics & {
  profiles?: Record<string, CoreMetrics | null>;
  current_profile?: string;
  profile_trial?: number;
  profile_trials_target?: number;
  experiment_complete?: boolean;
};

type PolicyChoice = { choice: number; p: number };

type Telemetry = {
  type: 'telemetry';
  trial: number;
  target: number;
  answer: number;
  correct: boolean;
  reward: number;
  expression?: string;
  task?: string;
  tokens?: string[];
  evaluation_profile?: string;
  profile_trial?: number;
  profile_trials_target?: number;
  profile_index?: number;
  profile_count?: number;
  experiment_complete?: boolean;
  learning_enabled?: boolean;
  metrics?: SuccessMetrics;
  accuracy?: number | null;
  policy?: {
    top: PolicyChoice[];
    entropy: number;
    chosen_probability: number;
    target_probability: number;
  };
  activity: [number, number][];
  plasticity: {
    mean_delta_w: number;
    active_synapses: number;
  };
};

const PROFILE_LABELS: Record<string, string> = {
  identity: 'Identity',
  successor: 'Successor',
  addition_seen: 'Addition · seen pair',
  addition_commutativity: 'Addition · reversed order',
  addition_heldout_pairs: 'Addition · unseen pair',
};

const PROFILE_DOM: Record<string, { exact: HTMLElement; balanced: HTMLElement; mae: HTMLElement }> = {
  identity: {
    exact: document.querySelector<HTMLElement>('#profile-identity-exact')!,
    balanced: document.querySelector<HTMLElement>('#profile-identity-balanced')!,
    mae: document.querySelector<HTMLElement>('#profile-identity-mae')!,
  },
  successor: {
    exact: document.querySelector<HTMLElement>('#profile-successor-exact')!,
    balanced: document.querySelector<HTMLElement>('#profile-successor-balanced')!,
    mae: document.querySelector<HTMLElement>('#profile-successor-mae')!,
  },
  addition_seen: {
    exact: document.querySelector<HTMLElement>('#profile-seen-exact')!,
    balanced: document.querySelector<HTMLElement>('#profile-seen-balanced')!,
    mae: document.querySelector<HTMLElement>('#profile-seen-mae')!,
  },
  addition_commutativity: {
    exact: document.querySelector<HTMLElement>('#profile-comm-exact')!,
    balanced: document.querySelector<HTMLElement>('#profile-comm-balanced')!,
    mae: document.querySelector<HTMLElement>('#profile-comm-mae')!,
  },
  addition_heldout_pairs: {
    exact: document.querySelector<HTMLElement>('#profile-heldout-exact')!,
    balanced: document.querySelector<HTMLElement>('#profile-heldout-balanced')!,
    mae: document.querySelector<HTMLElement>('#profile-heldout-mae')!,
  },
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
const glowWarm = new THREE.Color('#fff4d6');
const MAX_GLOW_POINTS = 1600;

let pointCloud: THREE.Points | null = null;
let glowCloud: THREE.Points | null = null;
let colors: Float32Array | null = null;
let basePositions: Float32Array | null = null;
let glowPositions: Float32Array | null = null;
let glowColors: Float32Array | null = null;
let neurons: Neuron[] = [];
let previousActiveIndices: number[] = [];
let layoutSummary = 'loading layout';
const byId = new Map<number, number>();

const answerEl = document.querySelector<HTMLElement>('#answer')!;
const targetEl = document.querySelector<HTMLElement>('#target')!;
const trialEl = document.querySelector<HTMLElement>('#trial')!;
const trialKindEl = document.querySelector<HTMLElement>('#trial-kind')!;
const plasticityEl = document.querySelector<HTMLElement>('#plasticity')!;
const accuracyOverallEl = document.querySelector<HTMLElement>('#accuracy-overall')!;
const accuracy20El = document.querySelector<HTMLElement>('#accuracy-20')!;
const accuracy100El = document.querySelector<HTMLElement>('#accuracy-100')!;
const accuracy500El = document.querySelector<HTMLElement>('#accuracy-500')!;
const accuracyCountEl = document.querySelector<HTMLElement>('#accuracy-count')!;
const balancedAccuracyEl = document.querySelector<HTMLElement>('#balanced-accuracy')!;
const maeEl = document.querySelector<HTMLElement>('#one-two-accuracy')!;
const withinOneEl = document.querySelector<HTMLElement>('#probe-balanced')!;
const taskEl = document.querySelector<HTMLElement>('#probe-one-two')!;
const stimulusEl = document.querySelector<SVGSVGElement>('#stimulus-view')!;
const policyEls = [0, 1, 2].map((i) => document.querySelector<HTMLElement>(`#policy-${i}`)!);
const policyLabels = [0, 1, 2].map((i) => document.querySelector<HTMLElement>(`#policy-label-${i}`)!);
const policyBars = [0, 1, 2].map((i) => document.querySelector<HTMLElement>(`#policy-bar-${i}`)!);
const statusEl = document.querySelector<HTMLElement>('#status')!;

function formatRate(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return value == null ? '—' : value.toFixed(digits);
}

function setNeuronColor(index: number, activity: number) {
  if (!colors) return;
  const neuron = neurons[index];
  const source = regionBase[neuron.region] ?? fallbackColor;
  const resting = source.clone().multiplyScalar(0.24);
  const contrast = Math.pow(Math.max(0, Math.min(1, activity)), 1.45);
  const mixed = resting.clone().lerp(hotColor, contrast);
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

function updateGlow(activityPairs: [number, number][]) {
  if (!glowCloud || !glowPositions || !glowColors || !basePositions) return;
  let count = 0;
  for (const [id, activity] of activityPairs) {
    if (activity < 0.5 || count >= MAX_GLOW_POINTS) continue;
    const index = byId.get(id);
    if (index === undefined) continue;
    const src = index * 3;
    const dst = count * 3;
    glowPositions[dst] = basePositions[src];
    glowPositions[dst + 1] = basePositions[src + 1];
    glowPositions[dst + 2] = basePositions[src + 2];
    const intensity = Math.pow(activity, 1.2);
    const color = (regionBase[neurons[index].region] ?? fallbackColor).clone().lerp(glowWarm, intensity);
    glowColors[dst] = color.r;
    glowColors[dst + 1] = color.g;
    glowColors[dst + 2] = color.b;
    count += 1;
  }
  (glowCloud.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
  (glowCloud.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  glowCloud.geometry.setDrawRange(0, count);
}

function renderExpression(expression: string | undefined) {
  while (stimulusEl.firstChild) stimulusEl.removeChild(stimulusEl.firstChild);
  const ns = 'http://www.w3.org/2000/svg';
  const background = document.createElementNS(ns, 'rect');
  background.setAttribute('x', '1');
  background.setAttribute('y', '1');
  background.setAttribute('width', '98');
  background.setAttribute('height', '98');
  background.setAttribute('rx', '8');
  background.setAttribute('class', 'stimulus-bg');
  stimulusEl.appendChild(background);

  const text = document.createElementNS(ns, 'text');
  text.setAttribute('x', '50');
  text.setAttribute('y', '56');
  text.setAttribute('text-anchor', 'middle');
  text.setAttribute('font-size', expression && expression.length > 10 ? '13' : '18');
  text.setAttribute('font-weight', '700');
  text.setAttribute('fill', '#f5f7fb');
  text.textContent = expression ?? '—';
  stimulusEl.appendChild(text);
}

function updatePolicy(policy: Telemetry['policy']) {
  if (!policy) return;
  for (let i = 0; i < 3; i += 1) {
    const item = policy.top[i];
    if (!item) {
      policyLabels[i].textContent = `#${i + 1}`;
      policyEls[i].textContent = '—';
      policyBars[i].style.width = '1%';
      continue;
    }
    policyLabels[i].textContent = `${item.choice}`;
    policyEls[i].textContent = `${(item.p * 100).toFixed(1)}%`;
    policyBars[i].style.width = `${Math.max(1, item.p * 100)}%`;
  }
}

function updateProfileResults(metrics: SuccessMetrics) {
  const profiles = metrics.profiles ?? {};
  for (const [profile, els] of Object.entries(PROFILE_DOM)) {
    const result = profiles[profile];
    els.exact.textContent = formatRate(result?.overall);
    els.balanced.textContent = formatRate(result?.balanced_accuracy);
    els.mae.textContent = formatNumber(result?.mean_absolute_error);
  }
}

function updateMetrics(metrics: SuccessMetrics | undefined, fallback: number | null | undefined, task: string | undefined) {
  if (!metrics) {
    accuracyOverallEl.textContent = formatRate(fallback);
    return;
  }
  accuracyOverallEl.textContent = formatRate(metrics.overall);
  accuracy20El.textContent = formatRate(metrics.recent_20);
  accuracy100El.textContent = formatRate(metrics.recent_100);
  accuracy500El.textContent = formatRate(metrics.recent_500);
  accuracyCountEl.textContent = `${metrics.successes.toLocaleString()} / ${metrics.attempts.toLocaleString()}`;
  balancedAccuracyEl.textContent = formatRate(metrics.balanced_accuracy);
  maeEl.textContent = formatNumber(metrics.mean_absolute_error);
  withinOneEl.textContent = formatRate(metrics.within_one_accuracy);
  taskEl.textContent = task ?? '—';
  updateProfileResults(metrics);
}

async function loadLayout() {
  const response = await fetch('http://localhost:8000/api/layout');
  if (!response.ok) throw new Error(`Layout request failed: ${response.status}`);
  const data = (await response.json()) as LayoutResponse;
  neurons = data.neurons;
  basePositions = new Float32Array(neurons.length * 3);
  colors = new Float32Array(neurons.length * 3);

  neurons.forEach((neuron, index) => {
    byId.set(neuron.id, index);
    const p = index * 3;
    basePositions![p] = neuron.x;
    basePositions![p + 1] = neuron.y;
    basePositions![p + 2] = neuron.z;
  });
  initializeColors();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(basePositions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    size: neurons.length > 5000 ? 0.06 : 0.09,
    vertexColors: true,
    transparent: true,
    opacity: 0.78,
    sizeAttenuation: true,
  });
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  glowPositions = new Float32Array(MAX_GLOW_POINTS * 3);
  glowColors = new Float32Array(MAX_GLOW_POINTS * 3);
  const glowGeometry = new THREE.BufferGeometry();
  glowGeometry.setAttribute('position', new THREE.BufferAttribute(glowPositions, 3));
  glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowColors, 3));
  glowGeometry.setDrawRange(0, 0);
  const glowMaterial = new THREE.PointsMaterial({
    size: neurons.length > 5000 ? 0.135 : 0.16,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });
  glowCloud = new THREE.Points(glowGeometry, glowMaterial);
  scene.add(glowCloud);

  const total = data.total_connectome_neurons;
  const coordinateLabel = total
    ? `${data.count.toLocaleString()} FlyWire coordinates / ${total.toLocaleString()} neurons`
    : `${data.count.toLocaleString()} neurons`;
  layoutSummary = `${coordinateLabel} · ${data.source}`;
  statusEl.textContent = layoutSummary;
}

function connectTelemetry() {
  const socket = new WebSocket('ws://localhost:8000/ws/telemetry');
  socket.addEventListener('open', () => {
    statusEl.textContent = `${layoutSummary} · preparing Stage 3S 120k symbolic curriculum/checkpoint…`;
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
    updateGlow(frame.activity);
    (pointCloud.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;

    renderExpression(frame.expression);
    updatePolicy(frame.policy);
    answerEl.textContent = `${frame.answer} ${frame.correct ? '✓' : '✕'}`;
    targetEl.textContent = `${frame.target}`;

    const profile = frame.evaluation_profile ?? frame.metrics?.current_profile ?? 'unknown';
    const profileLabel = PROFILE_LABELS[profile] ?? profile;
    const pTrial = frame.profile_trial ?? frame.metrics?.profile_trial ?? 0;
    const pTarget = frame.profile_trials_target ?? frame.metrics?.profile_trials_target ?? 0;
    trialEl.textContent = pTarget ? `${pTrial.toLocaleString()} / ${pTarget.toLocaleString()}` : frame.trial.toLocaleString();
    trialKindEl.textContent = `${profileLabel} · FROZEN`;
    plasticityEl.textContent = frame.learning_enabled === false
      ? 'frozen · no update'
      : `${frame.plasticity.active_synapses} · Δw ${frame.plasticity.mean_delta_w}`;
    updateMetrics(frame.metrics, frame.accuracy, frame.task);

    if (frame.experiment_complete || frame.metrics?.experiment_complete) {
      statusEl.textContent = `${layoutSummary} · Stage 3S COMPLETE · all symbolic generalization profiles saved`;
    } else {
      statusEl.textContent = `${layoutSummary} · Stage 3S ${profileLabel} (${frame.profile_index ?? '—'}/${frame.profile_count ?? 5}) · plasticity off · activity overlay is a proxy`;
    }
  });

  socket.addEventListener('close', () => {
    statusEl.textContent = `${layoutSummary} · telemetry disconnected · retrying…`;
    window.setTimeout(connectTelemetry, 1500);
  });
  socket.addEventListener('error', () => socket.close());
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
}
window.addEventListener('resize', resize);

function animate() {
  controls.update();
  if (pointCloud) pointCloud.rotation.y += 0.00035;
  if (glowCloud && pointCloud) glowCloud.rotation.copy(pointCloud.rotation);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

loadLayout().then(connectTelemetry).catch((error) => {
  console.error(error);
  statusEl.textContent = 'backend unavailable';
});
animate();
