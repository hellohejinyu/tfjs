const CliProgress = require('cli-progress')
const createCsvWriter = require('csv-writer').createObjectCsvWriter
const fs = require('fs')
const path = require('path')
const tf = require('@tensorflow/tfjs-node')

const Bar = new CliProgress.SingleBar({}, CliProgress.Presets.shades_classic)

const loadModel = async () => {
  const modelPath = 'file://' + path.join(__dirname, './mobilenet/model.json')
  try {
    const model = await tf.loadLayersModel(modelPath)
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

const readFileList = () => {
  const imagePath = path.resolve(__dirname, './images')
  const res = fs.readdirSync(imagePath)
  return res.filter(i => i.includes('.jpg') || i.includes('.jpeg')).map((f) => {
    return {
      name: f,
      path: path.join(imagePath, f)
    }
  })
}

const run = async () => {
  const model = await loadModel()
  if (model) {
    const fileList = readFileList()
    const recordFileName = `rec_${Date.now()}.csv`
    const csvWriter = createCsvWriter({
      path: path.resolve(__dirname, `./records/${recordFileName}`),
      header: [
        { id: 'name', title: 'IMAGE' },
        { id: 'mel', title: 'MEL' },
        { id: 'nv', title: 'NV' },
        { id: 'bcc', title: 'BCC' },
        { id: 'akiec', title: 'AKIEC' },
        { id: 'bkl', title: 'BKL' },
        { id: 'df', title: 'DF' },
        { id: 'vasc', title: 'VASC' }
      ]
    })
    Bar.start(fileList.length, 0)
    const records = []
    fileList.forEach((file, index) => {
      const tensor = preprocessImage(
        fs.readFileSync(file.path)
      )
      const prediction = model.predict(tensor).dataSync()
      records.push({
        name: file.name,
        mel: prediction[0],
        nv: prediction[1],
        bcc: prediction[2],
        akiec: prediction[3],
        bkl: prediction[4],
        df: prediction[5],
        vasc: prediction[6]
      })
      Bar.update(index + 1)
    })
    Bar.stop()
    try {
      await csvWriter.writeRecords(records)
      console.log(`${recordFileName} 文件写入成功！`)
    } catch (error) {
      console.log('csv 文件写入失败')
    }
  }
}

run()
