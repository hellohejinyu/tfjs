const CliProgress = require('cli-progress')
const cp = require('child_process')
const createCsvWriter = require('csv-writer').createObjectCsvWriter
const fs = require('fs')
const path = require('path')

process.env.TF_CPP_MIN_LOG_LEVEL = '2'

/** 多进程并行处理，可以更充分地利用多核 CPU 运算能力 */
const MAX_THREAD = 2

const Bar = new CliProgress.SingleBar({}, CliProgress.Presets.shades_classic)
let allRecords, progress

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

const triggerMsg = (i, { index, records }, finishCallback) => {
  progress[i] = index
  Bar.update(progress.reduce((prev, cur) => prev + cur), 0)
  if (records) {
    allRecords[i] = records
    if (allRecords.every(i => !!i)) {
      finishCallback()
    }
  }
}

const run = async () => {
  allRecords = new Array(MAX_THREAD).fill(null)
  progress = new Array(MAX_THREAD).fill(0)
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
  const fileList = readFileList()

  const partArr = new Array(MAX_THREAD)

  fileList.forEach((file, i) => {
    if (!partArr[i % MAX_THREAD]) {
      partArr[i % MAX_THREAD] = [file]
    } else {
      partArr[i % MAX_THREAD].push(file)
    }
  })

  Bar.start(fileList.length, 0)
  partArr.forEach((part, i) => {
    const child = cp.fork(path.join(__dirname, './child.js'))
    child.send(part)
    child.on('message', (msg) => {
      if (msg.records) {
        child.kill('SIGINT')
      }
      triggerMsg(i, msg, () => {
        Bar.stop()
        let normalizeRecords = []
        for (const rec of allRecords) {
          normalizeRecords = normalizeRecords.concat(rec)
        }
        csvWriter.writeRecords(normalizeRecords).then(() => {
          console.log(`${recordFileName} 文件写入成功！`)
        }).catch(() => {
          console.log('csv 文件写入失败')
        })
      })
    })
  })
}

run()
