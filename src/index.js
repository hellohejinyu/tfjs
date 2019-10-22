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
      Bar.update(index + 1)
      if (index < fileList.length - 1) {
        preprocessFile(index + 1)
      } else {
        Bar.stop()
        csvWriter.writeRecords(records).then(() => {
          console.log(`${recordFileName} 文件写入成功！`)
        }).catch(() => {
          console.log('csv 文件写入失败')
        })
      }
    }
    preprocessFile()
  }
}

run()
