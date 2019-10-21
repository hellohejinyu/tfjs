const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const path = require('path')

const loadModel = async () => {
  const modelPath = 'file://' + path.join(__dirname, './mobilenet/model.json')
  try {
    const model = await tf.loadLayersModel(modelPath)
    console.log('Load model successfully!')
    return model
  } catch (error) {
    console.log(error)
  }
}

const preprocessImage = (data) => {
  const tensor = tf.node.decodeJpeg(data)
    .resizeNearestNeighbor([224, 224])
    .toFloat()

  const offset = tf.scalar(127.5)
  return tensor.sub(offset)
    .div(offset)
    .expandDims()
}

const run = async () => {
  const model = await loadModel()
  if (model) {
    const tensor = preprocessImage(
      fs.readFileSync(path.resolve(__dirname, './images/1.jpg'))
    )
    const prediction = await model.predict(tensor).data()
    console.log(prediction)
  }
}

run()
