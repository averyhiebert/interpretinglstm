''' Some functions for creating text (HTML) with a background colour 
varying according to some statistic associated with the text. 

Tends to produce fairly large files.

This was not actually used for anything in the paper proper, but it was used
to some extent in one of the visualizations on the poster I presented.'''

import dominate.tags as dt
from colour import Color
import colorsys
import random

def deal_with_newlines(text):
    ''' Add a line break for newline characters. '''
    if "\n" not in text:
        return text
    else:
        chunks = [dt.span(t) for t in text.split("\n")]
        tag_list = []
        for i, chunk in enumerate(chunks):
            tag_list.append(chunk)
            if i != len(chunks) - 1:
                tag_list.append(dt.br())
        return tag_list

def tweak_whitespace(character):
    ''' Replace some whitespace characters to make them visible.
    Also, add a space after other characters for formatting reasons.
    
    (There would be space between the characters anyway, but including it
    explicitly makes the background colour continuous, 
    when there would be gaps otherwise.)'''
    if character==" ":
        return "_ "
    elif character=="\n":
        return "\\n\n"
    else:
        return character + " "

def style_string(bg=(1,1,1),fg=(0,0,0)):
    ''' Create a string specifying the given foreground and background colour
    in CSS syntax.  Also set text size.
    
    This is one reason why the file size is so big.  Really,
    the text size and foreground colour should not be set for each character
    individually.  Sorry.'''
    bgc = Color(rgb=bg)
    fgc = Color(rgb=fg)
    style = "background-color:"+ bgc.hex_l + ";color:" + fgc.hex_l
    style += ";font-size:10px"
    return style

def colored_text(texts,titles=None):
    ''' Texts should be a list of tuples [(string, value), ...] 
    
    Value should be a value between -1 and 1, or an (r,g,b) tuple.
    
    If all the strings begin or end with a space, the background colour will
    be more contiguous.
    
    Return a dominate.div object'''
    outer = dt.div()
    for i, (t, v) in enumerate(texts):
        c = v if type(v) == type((1,1,1)) else color_scale(v)
        text = deal_with_newlines(t)
        if titles:
            span = dt.span(text,style=style_string(bg=c),title=titles[i])
        else:
            span = dt.span(text,style=style_string(bg=c))
        outer.add(span)
    return outer

def color_scale(num):
    ''' Convert a number in (-1,1) to a colour.  

    Blue for negative, red for positive. '''
    positive_colour = Color("red")
    negative_colour = Color("blue")
    c = None
    if num >= 0:
        c = Color("red")
    else:
        num = num * (-1)
        c = Color("blue")

    # Clip 1 or -1 if necessary.
    if num > 1:
        num = 1
    elif num < -1:
        num = -1

    c.luminance = 1 - 0.3*num
    c.saturation = 1
    return c.rgb

def categorical_color_scale(n):
    ''' Return a list of n different colors (rgb tuples). '''
    hsvs = [(i*1.0/n, 0.5, 1) for i in range(n)]
    rgbs = [colorsys.hsv_to_rgb(*c) for c in hsvs]
    return rgbs

def get_tds(words, values):
    ''' Create a list of tds element containing coloured text.
    (Used later when creating multiple taples, separated by Details tags). '''
    tds = []
    for i in range(len(values[0])):
        # For each cell dimension, generate coloured text.
        vals = values[:,i]

        # Create the html
        words = [tweak_whitespace(w) for w in words] # for appearance
        data = [(word, vals[i]) for i, word in enumerate(words[:-1])]
        td = dt.td()
        td.add(colored_text(data))
        tds.append(td)
    return tds

def make_viz_table(words,datasets, label="Cell "):
    ''' Create a bunch of details tags, each of which contains a table,
    in order to display visualizations of multiple measurements
    (e.g. hidden state and cell state) side-by-side for each cell. '''
    td_lists = [get_tds(words,d) for d in datasets]
    td_lists = list(zip(*td_lists))

    outer = dt.div()
    for i in range(len(td_lists)):
        details = dt.details()
        details.add(dt.summary(label + str(i)))
        details.add(dt.table(dt.tr(*td_lists[i])))
        outer.add(details)
    return outer



if  __name__=="__main__":
    # Just a brief test.
    texts = [("test ",1-random.random()*2) for i in range(20)]

    with open("temp.html","w") as f:
        f.write(str(colored_text(texts)))
