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