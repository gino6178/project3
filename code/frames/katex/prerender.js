// Render every \( \) and $$ $$ in an HTML file with KaTeX at build time, so the page and its printout need no script.
// needs the katex package: npm install katex@0.16.9 (or NODE_PATH pointing at one)
const fs=require('fs'), katex=require('katex'); const [src,dst]=process.argv.slice(2);
let s=fs.readFileSync(src,'utf8'); let n=0, bad=0;
const dec=t=>t.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&nbsp;/g,' ').replace(/&#8217;/g,"'");
function r(tex,display){ try{ n++; return katex.renderToString(dec(tex),{displayMode:display,throwOnError:true,output:'html'}); }catch(e){ bad++; console.error('KaTeX:',e.message.slice(0,90)); return katex.renderToString(dec(tex),{displayMode:display,throwOnError:false,output:'html'}); } }
// leave <pre> (BibTeX) and <script> alone
const keep=[]; s=s.replace(/<(pre|script)[\s\S]*?<\/\1>/g,m=>{keep.push(m);return `\u0000${keep.length-1}\u0000`;});
s=s.replace(/\$\$([\s\S]+?)\$\$/g,(m,t)=>r(t,true));
s=s.replace(/\\\(([\s\S]+?)\\\)/g,(m,t)=>r(t,false));
s=s.replace(/\u0000(\d+)\u0000/g,(m,i)=>keep[+i]);
// the auto-render scripts are no longer needed; the stylesheet stays
s=s.replace(/<script[^>]*katex[^>]*><\/script>\s*/g,'').replace(/<script[^>]*auto-render[\s\S]*?<\/script>\s*/g,'');
fs.writeFileSync(dst,s); console.log('rendered',n,'expressions,',bad,'with errors');
