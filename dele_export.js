// Eksporterer alle deles former fra index.html til tools/dele.json (bruges af skabeloner.py)
const fs=require("fs"), path=require("path");
const html=fs.readFileSync(path.join(__dirname,"..","index.html"),"utf8");
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join("\n");
const code=scripts.split("/* Klikbare former */")[0];
const f=new Function("window","document",code+"\nreturn {PARTS,G};");
const {PARTS,G}=f({}, {querySelector:()=>null,querySelectorAll:()=>[]});
const o={}; PARTS.forEach(p=>o[p.id]={g:p.g,navn:p.navn,lag:G[p.g].lag,s:p.shape});
fs.writeFileSync(path.join(__dirname,"dele.json"),JSON.stringify(o,null,0));
console.log(Object.keys(o).length,"dele");
