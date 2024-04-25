import mongoose from "mongoose";
import {makeApiRecord} from "./model/api-record.js";
import {makeMashupRecord} from "./model/mashup-record.js";
import {load_data} from "./load-api-data.js";

const connection_uri = 'mongodb://localhost:27017/';
const collection_name = 'pa04';
const connection = await mongoose.connect(connection_uri + collection_name);

export const ApiRecord = makeApiRecord(connection);
export const MashupRecord = makeMashupRecord(connection);


export async function load_data_to_db() {
    await ApiRecord.collection.drop();
    console.log('DROPPED THE API COLLECTION');

    await MashupRecord.collection.drop();
    console.log('DROPPED THE MASHUP COLLECTION');

    load_data(`${process.cwd()}/data/api.txt`, ApiRecord).then(() => {
        console.log('LOADED API RECORDS TO DB');
    });
    load_data(`${process.cwd()}/data/mashup.txt`, MashupRecord).then(() => {
        console.log('LOADED MASHUP RECORDS TO DB');
    });
}

await load_data_to_db();

