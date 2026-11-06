import { useEffect, useRef, useState } from 'react';

export type FieldPlane = {
  description: string; component: string;
  horizontal_axis: string; vertical_axis: string;
  horizontal_um: number[]; vertical_um: number[]; values: number[][];
  source_shape: number[]; peak_abs: number; time_um_c?: number; wavelength_um?: number;
};

export function CalibrationField({field}: {field: FieldPlane}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const {horizontal_um: xs, vertical_um: ys, values, peak_abs: peak} = field;
  const [cursor, setCursor] = useState(() => [Math.floor(xs.length / 2), Math.floor(ys.length / 2)]);
  const ix = Math.min(cursor[0], xs.length - 1), iy = Math.min(cursor[1], ys.length - 1);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const pixels = context.createImageData(xs.length, ys.length);
    for (let y = 0; y < ys.length; y++) {
      for (let x = 0; x < xs.length; x++) {
        const u = peak ? values[y][x] / peak : 0;
        const magnitude = Math.min(1, Math.abs(u));
        const color = u >= 0 ? [255, 188, 100] : [82, 203, 255];
        const offset = 4 * ((ys.length - 1 - y) * xs.length + x);
        for (let c = 0; c < 3; c++) pixels.data[offset + c] = 14 + magnitude * (color[c] - 14);
        pixels.data[offset + 3] = 255;
      }
    }
    context.putImageData(pixels, 0, 0);
  }, [xs, ys, values, peak]);
  const rangeX = xs[xs.length - 1] - xs[0], rangeY = ys[ys.length - 1] - ys[0];
  return <section aria-label="Measured field plane" className="cal-field">
    <h3>Measured field</h3>
    <p>{field.description}{field.wavelength_um != null && ` · ${field.wavelength_um} µm`}
      {field.time_um_c != null && ` · t = ${field.time_um_c.toFixed(2)} µm/c`}</p>
    <canvas ref={canvas} width={xs.length} height={ys.length} role="img" tabIndex={0}
      aria-label={`${field.component} field. Use arrow keys to inspect neighboring samples.`}
      style={{aspectRatio: String(rangeX / rangeY)}}
      onKeyDown={event => {
        const directions: Record<string, number[]> = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]};
        const delta = directions[event.key];
        if (!delta) return;
        event.preventDefault();
        setCursor([Math.max(0, Math.min(xs.length-1, ix+delta[0])), Math.max(0, Math.min(ys.length-1, iy+delta[1]))]);
      }}
      onClick={event => {
        const box = event.currentTarget.getBoundingClientRect();
        const x = Math.max(0, Math.min(xs.length-1, Math.floor((event.clientX-box.left)/box.width*xs.length)));
        const y = Math.max(0, Math.min(ys.length-1, ys.length-1-Math.floor((event.clientY-box.top)/box.height*ys.length)));
        setCursor([x,y]);
      }}/>
    <p className="cal-field-scale">−{peak.toPrecision(3)} <span/> +{peak.toPrecision(3)} · solver units</p>
    <p><output aria-label="Field sample">{field.horizontal_axis} = {xs[ix].toFixed(4)} µm · {field.vertical_axis} = {ys[iy].toFixed(4)} µm · {field.component} = {values[iy][ix].toPrecision(7)}</output></p>
  </section>;
}
