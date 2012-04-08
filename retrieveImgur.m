function [image] = retrieveImgur()
images = ['http://i.imgur.com/QHXMZ.jpg';
        'http://i.imgur.com/DL6g2.jpg';
        'http://i.imgur.com/u1nEqh.jpg';
        'http://i.imgur.com/9Mo6ah.jpg';
        'http://i.imgur.com/z1mCV.jpg';
        'http://i.imgur.com/eTYbp.jpg';
        'http://i.imgur.com/2aiAx.jpg';
        'http://i.imgur.com/pVsE7.gif';
        'http://i.imgur.com/ZAFcmh.jpg';
        'http://i.imgur.com/8iHbJ.jpg';
        'http://i.imgur.com/RumPMh.jpg';
        'http://i.imgur.com/rCbLXh.jpg';
        'http://i.imgur.com/f2dJGh.jpg';
        'http://i.imgur.com/YyPti.jpg';
        'http://i.imgur.com/KUc8fh.jpg';
        'http://i.imgur.com/TxTS9h.jpg';
        'http://i.imgur.com/k8Skoh.jpg';
        'http://i.imgur.com/REcqmh.jpg';
        'http://i.imgur.com/jgv8M.jpg';
        'http://i.imgur.com/vFxN2.jpg'];

image = rgb2gray(imread(images(1)));