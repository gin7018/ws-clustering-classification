import mongoose from "mongoose";

export function makeMashupRecord(db) {
    return db.model('MashupRecord', new mongoose.Schema({
        id: {type: String},
        title: {type: String},
        summary: {type: String},
        rating: {type: Number, default: 0.0},
        name: {type: String},
        label: {type: String},
        author: {type: String},
        description: {type: String},
        type: {type: String},
        downloads : {type: Number, default: 0.0},
        useCount : {type: Number, default: 0.0},
        sampleUrl: {type: String},
        dateModified: {type: String},
        numComments : {type: Number, default: 0.0},
        commentsUrl: {type: String},
        tags: [{type: String}],
        apis: [{type: ApiSchema}],
        updated: {type: Date}
    }));
}

const ApiSchema = new mongoose.Schema({
    name: {type: String},
    link: {type: String}
})