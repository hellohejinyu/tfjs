const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const path = require('path')

const loadModel = async () => {
  const modelPath = 'file://' + path.join(__dirname, './mobilenet/model.json')
  try {
    const model = await tf.loadLayersModel(modelPath)
    return model
  } catch (error) {
    console.log(error)
  }
}

const offset = tf.scalar(127.5)

const preprocessImage = (filePath) => tf.tidy(() => {
  const data = fs.readFileSync(filePath)
  const tensor = tf.node.decodeJpeg(data)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
  return tensor.sub(offset)
    .div(offset)
    .expandDims()
})

const run = async (fileList) => {
  if (!fileList.length) {
    return
  }
  const model = await loadModel()
  const records = []
  const preprocessFile = (index = 0) => {
    const tensor = preprocessImage(fileList[index].path)
    const prediction = model.predict(tensor)
    const data = prediction.dataSync()
    tensor.dispose()
    prediction.dispose()
    records.push({
      name: fileList[index].name,
      mel: data[0],
      nv: data[1],
      bcc: data[2],
      akiec: data[3],
      bkl: data[4],
      df: data[5],
      vasc: data[6]
    })
    if (index < fileList.length - 1) {
      process.send({ index: index + 1 })
      preprocessFile(index + 1)
    } else {
      process.send({ index: fileList.length, records: records })
    }
  }
  preprocessFile()
}

process.on('message', (fileList) => {
  run(fileList)
})
