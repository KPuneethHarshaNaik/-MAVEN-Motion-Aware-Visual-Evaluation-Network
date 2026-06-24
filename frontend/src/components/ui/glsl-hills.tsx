import { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface GLSLHillsProps {
  className?: string;
}

const GLSLHills = ({ className = '' }: GLSLHillsProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const w = container.clientWidth;
    const h = container.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(0, -60, 120);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(renderer.domElement);

    const uniforms = { time: { value: 0 } };

    const vertexShader = `
      varying vec2 vUv;
      varying float vElevation;
      uniform float time;

      vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
      float snoise(vec2 v){
        const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
        vec2 i = floor(v + dot(v, C.yy));
        vec2 x0 = v - i + dot(i, C.xx);
        vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz;
        x12.xy -= i1;
        i = mod(i, 289.0);
        vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
        vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
        m = m*m; m = m*m;
        vec3 x_ = 2.0 * fract(p * C.www) - 1.0;
        vec3 h = abs(x_) - 0.5;
        vec3 ox = floor(x_ + 0.5);
        vec3 a0 = x_ - ox;
        m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
        vec3 g;
        g.x = a0.x * x0.x + h.x * x0.y;
        g.yz = a0.yz * x12.xz + h.yz * x12.yw;
        return 130.0 * dot(m, g);
      }

      void main() {
        vUv = uv;
        vec4 mp = modelMatrix * vec4(position, 1.0);
        float elev = snoise(vec2(mp.x * 0.02, mp.y * 0.02 + time * 0.3)) * 12.0;
        elev += snoise(vec2(mp.x * 0.06, mp.y * 0.06 + time * 0.15)) * 4.0;
        mp.z += elev;
        vElevation = elev;
        gl_Position = projectionMatrix * viewMatrix * mp;
      }
    `;

    const fragmentShader = `
      varying vec2 vUv;
      varying float vElevation;

      void main() {
        vec3 deep = vec3(0.02, 0.03, 0.08);
        vec3 mid = vec3(0.34, 0.22, 0.96);
        vec3 peak = vec3(0.02, 0.71, 0.83);

        float t = smoothstep(-12.0, 16.0, vElevation);
        vec3 col = mix(deep, mid, t);
        col = mix(col, peak, smoothstep(0.65, 1.0, t) * 0.6);

        // Grid lines
        vec2 g = fract(vUv * 50.0);
        float line = smoothstep(0.04, 0.0, g.x) + smoothstep(0.04, 0.0, g.y);
        col = mix(col, peak, line * 0.2 * t);

        // Edge fade
        float fade = 1.0 - smoothstep(0.25, 0.5, length(vUv - 0.5));
        gl_FragColor = vec4(col, fade * 0.85);
      }
    `;

    const geo = new THREE.PlaneGeometry(256, 256, 256, 256);
    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2.5;
    scene.add(mesh);

    let raf: number;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      uniforms.time.value += 0.005;
      renderer.render(scene, camera);
    };
    tick();

    const onResize = () => {
      if (!container) return;
      const nw = container.clientWidth;
      const nh = container.clientHeight;
      renderer.setSize(nw, nh);
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(raf);
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 1,
        pointerEvents: 'none',
      }}
    />
  );
};

export default GLSLHills;
