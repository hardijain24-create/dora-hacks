const http = require('http');
const fs = require('fs');

http.createServer((req, res) => {
  if (req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      fs.appendFileSync('scratch/browser.log', body + '\n');
      res.writeHead(200, { 'Access-Control-Allow-Origin': '*' });
      res.end('OK');
    });
  } else {
    res.writeHead(200, { 'Access-Control-Allow-Origin': '*' });
    res.end('OK');
  }
}).listen(4000, () => {
  console.log('Log server listening on port 4000');
});
