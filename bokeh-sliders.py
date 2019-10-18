""" Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve bokeh-sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
"""
import time
from tornado import gen
from functools import partial
from random import random
from threading import Thread

import imageio
import numpy as np
from PIL import Image
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.layouts import widgetbox
from bokeh.models import ColumnDataSource, Label
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from main import GeneticAlg
from shapes import Circle

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))



# Set up plot
sin_plot = figure(plot_height=400, plot_width=400, title="my sine wave",
                  tools="crosshair,pan,reset,save,wheel_zoom",
                  x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

sin_plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.05)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1)


# Set up callbacks
def update_title(attrname, old, new):
    sin_plot.title.text = text.value


text.on_change('value', update_title)


def update_data(attrname, old, new):
    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(k*x + w) + b

    source.data = dict(x=x, y=y)


for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(text, offset, amplitude, phase, freq)

a = np.array(imageio.imread('circles.jpg'))#[:,:,0]
# a = np.array([[1,2], [3, 4]])
kind_of_image_plot = figure(title="RGBA image",
                            plot_width=int(a.shape[1]/1),
                            plot_height=int(a.shape[0]/1),
                            # x_range = [0,10], y_range = [0,10],
                            )


curdoc().title = "Sliders"

# Open image, and make sure it's RGB*A*
circles = Image.open('circles.jpg').convert('RGBA')
circles = Image.open('mona-lisa.jpg!HalfHD.jpg').convert('RGBA')
xdim, ydim = circles.size
print("Dimensions: ({xdim}, {ydim})".format(**locals()))
# Create an array representation for the image `img`, and an 8-bit "4
# layer/RGBA" version of it `view`.
img = np.empty((ydim, xdim), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
# Copy the RGBA image into view, flipping it so it comes right-side up
# with a lower-left origin
view[:, :, :] = np.flipud(np.asarray(circles))

# Display the 32-bit RGBA image
dim = max(xdim, ydim)
image_figure = figure(title="Lena",
                      x_range=(0, dim), y_range=(0, dim),
                      # Specifying xdim/ydim isn't quire right :-(
                      # width=xdim, height=ydim,
                      )
label = Label(x=1.1, y=18, text=str("iteration"), text_font_size='15pt', text_color='#e5a5e5')
image_figure.add_layout(label)
im = image_figure.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)


opacity = Slider(title="opacity", value=100.0, start=1.0, end=255.0)


# Set up callbacks
@gen.coroutine
def update_image(attrname, new_image, new):
    print(attrname, np.sum(np.asarray(new_image)), new)

    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(new_image))
    new_data = dict()
    new_data['image'] = [img]
    im.data_source.data = new_data



opacity.on_change('value', update_image)
image_inputs = widgetbox(opacity)

page_layout = layout([[image_figure], [image_inputs], [sin_plot, inputs], ],
                     sizing_mode='scale_both')
curdoc().add_root(page_layout)

# this must only be modified from a Bokeh session callback
source = ColumnDataSource(data=dict(x=[0], y=[0]))

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

die = False


def run_alg():

    alg = GeneticAlg(40, shape=Circle, img=np.asarray(circles), num_iter=2000)
    for c_i in alg.run():
        x, y = random(), random()
        c_i = np.array(c_i, dtype=np.uint8)
        circ = Image.fromarray(c_i).convert('RGBA')
        doc.add_next_tick_callback(partial(update_image, 'value',
                                           new_image=circ, new=y))


try:
    thread = Thread(target=run_alg)
    thread.start()
except KeyboardInterrupt:
    # global die
    die = True
    raise KeyboardInterrupt

def cleanup_session(session_context):
    ''' This function is called when a session is closed. '''
    global die
    die = True
    thread.join(4)


doc.on_session_destroyed(cleanup_session)
