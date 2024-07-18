import * as dotenv from 'dotenv'
dotenv.config()
import fs from "fs"
import { ethers } from "ethers"
import lighthouse from '@lighthouse-web3/sdk'

const signAuthMessage = async (publicKey, privateKey) => {
  const provider = new ethers.JsonRpcProvider()
  const signer = new ethers.Wallet(privateKey, provider)
  const messageRequested = (await lighthouse.getAuthMessage(publicKey)).data.message
  const signedMessage = await signer.signMessage(messageRequested)
  return signedMessage
}

const decrypt = async (cid, fileName) => {
  const publicKey = "0x595a2C574Cd823F8504de02e3110670360A5e711" //Example: '0xa3c960b3ba29367ecbcaf1430452c6cd7516f588'
  const privateKey = process.env.PRIVATE_KEY

  // Get file encryption key
  const signedMessage = await signAuthMessage(publicKey, privateKey)
  const fileEncryptionKey = await lighthouse.fetchEncryptionKey(
    cid,
    publicKey,
    signedMessage
  )

  // Decrypt File
  const decrypted = await lighthouse.decryptFile(
    cid,
    fileEncryptionKey.data.key
  )

  // Save File
  fs.createWriteStream("./files/" + fileName).write(Buffer.from(decrypted))
}
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error("Please provide the path to the file as an argument.");
  process.exit(1);
}

const cid = args[0];
const filename = args[1];
decrypt(cid, filename)
