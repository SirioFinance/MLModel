import {ethers} from "ethers"
import * as dotenv from 'dotenv'
dotenv.config()
import lighthouse from '@lighthouse-web3/sdk'
import kavach from "@lighthouse-web3/kavach"

const signAuthMessage = async(privateKey) =>{
  const signer = new ethers.Wallet(privateKey)
  const authMessage = await kavach.getAuthMessage(signer.address)
  const signedMessage = await signer.signMessage(authMessage.message)
  const { JWT, error } = await kavach.getJWT(signer.address, signedMessage)
  return(JWT)
}

const uploadEncrypted = async(pathToFile) =>{
  /**
   * This function lets you upload a file to Lighthouse with encryption enabled.
   * 
   * @param {string} path - Location of your file.
   * @param {string} apiKey - Your unique Lighthouse API key.
   * @param {string} publicKey - User's public key for encryption.
   * @param {string} signedMessage - A signed message or JWT used for authentication at encryption nodes.
   * 
   * @return {object} - Returns details of the encrypted uploaded file.
   */
  
  const apiKey = 'bcd8b272.06cb4eac003b4d22827e4ed71fc93278'
  const publicKey = '0x595a2C574Cd823F8504de02e3110670360A5e711'
  const privateKey = process.env.PRIVATE_KEY
  
  const signedMessage = await signAuthMessage(privateKey)
  
  const response = await lighthouse.uploadEncrypted(pathToFile, apiKey, publicKey, signedMessage)
  console.log(response)
  /* Sample Response
  {
    data: [
      {
        Name: 'decrypt.js',
        Hash: 'QmeLFQxitPyEeF9XQEEpMot3gfUgsizmXbLha8F5DLH1ta',
        Size: '1198'
      }
    ]
  }
  */
}
const args = process.argv.slice(2);
if (args.length !== 1) {
  console.error("Please provide the path to the file as an argument.");
  process.exit(1);
}

const pathToFile = args[0];
uploadEncrypted(pathToFile);