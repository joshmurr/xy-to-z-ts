import * as tf from '@tensorflow/tfjs'

interface Mouse {
  x: number
  y: number
}

const XY2Z_PATH = 'assets/xy_to_z/model.json'
const DECODER_PATH = 'assets/decoder/model.json'

const in_canvas = <HTMLCanvasElement>document.getElementById('input')
const out_canvas = <HTMLCanvasElement>document.getElementById('output')
const ctx = in_canvas.getContext('2d')

let xy2z_model: tf.GraphModel
let decoder_model: tf.GraphModel
const mouse: Mouse = {
  x: 0,
  y: 0,
}

in_canvas.width = 512
in_canvas.height = 512

ctx.fillStyle = 'white'
ctx.fillRect(0, 0, 512, 512)

const init = async () => {
  xy2z_model = await tf.loadGraphModel(XY2Z_PATH)

  decoder_model = await tf.loadGraphModel(DECODER_PATH)
  const ret = decoder_model.predict(tf.zeros([1, 64])) as tf.Tensor
  ret.dispose()
}

const run_model = async () => {
  const { x, y } = normalize_coords()

  const logits = tf.tidy(() => {
    const input = tf.tensor([[x, y], [0]])
    const z = xy2z_model.predict(input) as tf.Tensor
    const decoded = decoder_model.predict(z) as tf.Tensor
    return decoded
  })
  //const data = await logits.data()
  const img_t = logits.slice(0, [1]).transpose([1, 2, 0]) as tf.Tensor3D
  tf.browser.toPixels(img_t, out_canvas)
}

const normalize_coords = () => {
  return {
    x: mouse.x / 256 - 1,
    y: mouse.y / 256 - 1,
  }
}

const get_xy = (e: MouseEvent) => {
  const target = e.target as Element
  const rect = target.getBoundingClientRect()
  mouse.x = e.clientX - rect.left //x position within the element.
  mouse.y = e.clientY - rect.top //y position within the element.

  run_model()
}

init()

in_canvas.addEventListener('mousemove', get_xy)
// in_canvas.addEventListener('click', run_model)
