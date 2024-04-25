import fs from "fs";


export async function load_data(file, Record) {
    try {
        const data = fs.readFileSync(file, 'utf8').split('\n');
        let schema_fields = Object.keys(Record.schema.paths);
        schema_fields = schema_fields.filter((field) => {return !['__id', '__v'].includes(field)})

        for (let i = 0; i < data.length; i++) {
            let line = data[i];
            let values = line.split('$#$');

            let record = new Record();
            schema_fields.forEach((field, index) => {
                try {
                    record[field] = processDataForType(values[index], Record.schema.path(field).instance);
                } catch (e) {
                    console.error(`[LOAD API DATA] status=failed; error=parser failed on ${field} = ${values[index]}`);
                    throw e;
                }
            })
            await record.save();
        }
        console.log(`[LOAD API DATA] status=success; collection=${Record.name}`);

    } catch (e) {
        console.error(`[LOAD API DATA] status=failed; error=${e.message}`);
    }
}

function processDataForType(data, type) {
    if (!data && type === 'Array') return [];
    if (!data) return null;

    if (type === 'String') {
        return data;
    }
    else if (type === 'Number') {
        return Number(data);
    }
    else if (type === 'Date') {
        return Date.parse(data);
    }
    else if (type === 'Array') {
        let entries = data.includes('###')? data.split('###'): data.split(',');
        entries = entries.filter(entry => entry.length > 0);

        if (entries[0].includes('$$$')) {
            return entries.map(entry => {
                let fields = entry.split('$$$');
                return {name: fields[0], link: fields[1]};
            });
        }
        return entries;
    }
    return data;
}