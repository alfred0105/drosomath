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

type CoreMetrics = {
  overall: number | null;
  balanced_accuracy?: number | null;
  one_vs_two_accuracy?: number | null;
  recent_20: number | null;
  recent_100: number | null;
  recent_500: number | null;
  successes: number;
  attempts: number;
  by_target_accuracy?: Record<string, number | null>;
  confusion_matrix?: number[][];
};

type SuccessMetrics = CoreMetrics & {
  probe?: CoreMetrics;
  profiles?: Record<string, CoreMetrics | null>;
  current_profile?: string;
  profile_trial?: number;
  profile_trials_target?: number;
  experiment_complete?: boolean;
};

type DotStimulus = {
  kind: 'dots';
  numerosity: number;
  dots: Array<{ x: number; y: number; r: number; gain: number }>;
  controls?: {
    stimulus_profile?: string;
    signal_energy: number;
    area_factor: number;
    requested_area_factor?: number;
    post_noise_energy: number;
    min_pair_distance?: number | null;
    gain_ratio?: number | null;
    fractional_position?: boolean;
    novel_area?: boolean;
    close_spacing?: boolean;
    strong_brightness?: boolean;
    rasterizer?: string;
    visual_preprocess?: string;
  };
};

type Telemetry = {
  type: 'telemetry';
  telemetry_source?: string;
  activity_source?: string;
  learning_model?: string;
  phase?: string;
  trial_kind?: string;
  learning_enabled?: boolean;
  trial: number;
  target: number;
  answer: number;
  correct: boolean;
  reward: number;
  evaluation_profile?: string;
  profile_trial?: number;
  profile_trials_target?: number;
  profile_index?: number;
  profile_count?: number;
  experiment_complete?: boolean;
  accuracy?: number | null;
  metrics?: SuccessMetrics;
  stimulus?: DotStimulus;
  policy?: {
    p0: number;
    p1: number;
    p2: number;
    entropy: number;
  };
  activity: [number, number][];
  plasticity: {
    mean_delta_w: number;
    active_synapses: number;
  };
};

const PROFILE_LABELS: Record<string, string> = {
  position_only: 'Position',
  brightness_only: 'Brightness',
  combined: 'Combined',
};

const PROFILE_DOM: Record<string, { balanced: HTMLElement; oneTwo: HTMLElement; two: HTMLElement }> = {
  position_only: {
    balanced: document.querySelector<HTMLElement>('#profile-position-balanced')!,
    oneTwo: document.querySelector<HTMLElement>('#profile-position-one-two')!,
    two: document.querySelector<HTMLElement>('#profile-position-two')!,
  },
  brightness_only: {
    balanced: document.querySelector<HTMLElement>('#profile-brightness-balanced')!,
    oneTwo: document.querySelector<HTMLElement>('#profile-brightness-one-two')!,
    two: document.querySelector<HTMLElement>('#profile-brightness-two')!,
  },
  combined: {
    balanced: document.querySelector<HTMLElement>('#profile-combined-balanced')!,
    oneTwo: document.querySelector<HTMLElement>('#profile-combined-one-two')!,
    two: document.querySelector<HTMLElement>('#profile-combined-two')!,
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
const oneTwoAccuracyEl = document.querySelector<HTMLElement>('#one-two-accuracy')!;
const probeBalancedEl = document.querySelector<HTMLElement>('#probe-balanced')!;
const probeOneTwoEl = document.querySelector<HTMLElement>('#probe-one-two')!;
const class0El = document.querySelector<HTMLElement>('#class-0')!;
const class1El = document.querySelector<HTMLElement>('#class-1')!;
const class2El = document.querySelector<HTMLElement>('#class-2')!;
const stimulusEl = document.querySelector<SVGSVGElement>('#stimulus-view')!;
const policyEls = [0, 1, 2].map((i) => document.querySelector<HTMLElement>(`#policy-${i}`)!);
const policyBarEls = [0, 1, 2].map((i) => document.querySelector<HTMLElement>(`#policy-bar-${i}`)!);
const statusEl = document.querySelector<HTMLElement>('#status')!;

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
    const color = (regionBase[neurons[index].region] ?? fallbackColor)
      .clone()
      .lerp(glowWarm, intensity);
    glowColors[dst] = color.r;
    glowColors[dst + 1] = color.g;
    glowColors[dst + 2] = color.b;
    count += 1;
  }

  const positionAttr = glowCloud.geometry.getAttribute('position') as THREE.BufferAttribute;
  const colorAttr = glowCloud.geometry.getAttribute('color') as THREE.BufferAttribute;
  positionAttr.needsUpdate = true;
  colorAttr.needsUpdate = true;
  glowCloud.geometry.setDrawRange(0, count);
}

function formatRate(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function updateProfileResults(metrics: SuccessMetrics) {
  const profiles = metrics.profiles ?? {};
  Object.entries(PROFILE_DOM).forEach(([profile, elements]) => {
    const result = profiles[profile];
    elements.balanced.textContent = formatRate(result?.balanced_accuracy);
    elements.oneTwo.textContent = formatRate(result?.one_vs_two_accuracy);
    elements.two.textContent = formatRate(result?.by_target_accuracy?.['2']);
  });
}

function updateMetrics(metrics: SuccessMetrics | undefined, fallbackAccuracy: number | null | undefined) {
  if (!metrics) {
    accuracyOverallEl.textContent = formatRate(fallbackAccuracy);
    return;
  }

  accuracyOverallEl.textContent = formatRate(metrics.overall);
  accuracy20El.textContent = formatRate(metrics.recent_20);
  accuracy100El.textContent = formatRate(metrics.recent_100);
  accuracy500El.textContent = formatRate(metrics.recent_500);
  accuracyCountEl.textContent = `${metrics.successes.toLocaleString()} / ${metrics.attempts.toLocaleString()}`;
  balancedAccuracyEl.textContent = formatRate(metrics.balanced_accuracy);
  oneTwoAccuracyEl.textContent = formatRate(metrics.one_vs_two_accuracy);

  const byTarget = metrics.by_target_accuracy ?? {};
  class0El.textContent = formatRate(byTarget['0']);
  class1El.textContent = formatRate(byTarget['1']);
  class2El.textContent = formatRate(byTarget['2']);

  const probe = metrics.probe;
  probeBalancedEl.textContent = formatRate(probe?.balanced_accuracy);
  probeOneTwoEl.textContent = formatRate(probe?.one_vs_two_accuracy);
  updateProfileResults(metrics);
}

function renderStimulus(stimulus: DotStimulus | undefined) {
  while (stimulusEl.firstChild) stimulusEl.removeChild(stimulusEl.firstChild);
  if (!stimulus) return;

  const ns = 'http://www.w3.org/2000/svg';
  const background = document.createElementNS(ns, 'rect');
  background.setAttribute('x', '1');
  background.setAttribute('y', '1');
  background.setAttribute('width', '98');
  background.setAttribute('height', '98');
  background.setAttribute('rx', '8');
  background.setAttribute('class', 'stimulus-bg');
  stimulusEl.appendChild(background);

  for (const dot of stimulus.dots) {
    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('cx', (dot.x * 100).toFixed(2));
    circle.setAttribute('cy', (dot.y * 100).toFixed(2));
    circle.setAttribute('r', (dot.r * 100).toFixed(2));
    circle.setAttribute('class', 'stimulus-dot');
    circle.setAttribute('opacity', Math.min(1, 0.48 + dot.gain * 0.38).toFixed(2));
    stimulusEl.appendChild(circle);
  }
}

function updatePolicy(policy: Telemetry['policy']) {
  if (!policy) return;
  const values = [policy.p0, policy.p1, policy.p2];
  values.forEach((value, index) => {
    policyEls[index].textContent = `${(value * 100).toFixed(1)}%`;
    policyBarEls[index].style.width = `${Math.max(1, value * 100)}%`;
  });
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
    statusEl.textContent = `${layoutSummary} · preparing Stage 2.2B robust 10k replay/checkpoint…`;
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

    const colorAttr = pointCloud.geometry.getAttribute('color') as THREE.BufferAttribute;
    colorAttr.needsUpdate = true;

    renderStimulus(frame.stimulus);
    updatePolicy(frame.policy);
    answerEl.textContent = `${frame.answer} ${frame.correct ? '✓' : '✕'}`;
    targetEl.textContent = `${frame.target}`;

    const profile = frame.evaluation_profile ?? frame.metrics?.current_profile ?? 'unknown';
    const profileLabel = PROFILE_LABELS[profile] ?? profile;
    const profileTrial = frame.profile_trial ?? frame.metrics?.profile_trial ?? 0;
    const profileTarget = frame.profile_trials_target ?? frame.metrics?.profile_trials_target ?? 0;
    trialEl.textContent = profileTarget
      ? `${profileTrial.toLocaleString()} / ${profileTarget.toLocaleString()}`
      : frame.trial.toLocaleString();
    trialKindEl.textContent = `${profileLabel} · ROBUST FROZEN`;

    plasticityEl.textContent = frame.learning_enabled === false
      ? 'frozen · no update'
      : `${frame.plasticity.active_synapses} · Δw ${frame.plasticity.mean_delta_w >= 0 ? '+' : ''}${frame.plasticity.mean_delta_w}`;
    updateMetrics(frame.metrics, frame.accuracy);

    if (frame.experiment_complete || frame.metrics?.experiment_complete) {
      statusEl.textContent = `${layoutSummary} · Stage 2.2B COMPLETE · all 3 frozen robust OOD profiles saved`;
    } else {
      statusEl.textContent = `${layoutSummary} · Stage 2.2B ${profileLabel} (${frame.profile_index ?? '—'}/${frame.profile_count ?? 3}) · robust encoder · plasticity off · activity overlay is a proxy`;
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
  if (glowCloud && pointCloud) glowCloud.rotation.copy(pointCloud.rotation);
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
