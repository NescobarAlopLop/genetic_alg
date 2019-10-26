
writing gifs, might be useful:

imageio.mimwrite('file.gif', frames, format='GIF-PIL', quantizer=0)

---
After that you need to open two terminals at app/, in the first one you need to run the bokeh server with the command:

```bash
bokeh serve ./bokeh-sliders.py --allow-websocket-origin=127.0.0.1:5000
```

In the second one run the command:

```
python hello.py
```

After this the webpage should 'just work' by opening a webpage to http://localhost:5000

data format

generation is a set of dnas
dna is a set of chromosomes
chromosome is a set of variables

size of chromosome num elements describing shape:
 - circle: r, x, y, r, g, b = 6 
 - triangle: x0, y0, x1, y1, x2, y2, r, g, b = 9 


