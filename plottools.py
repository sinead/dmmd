import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

"""
functions used in yearbook_qualitative_experiments.ipynb
"""

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        modified from Matplotlib example https://matplotlib.org/stable/gallery/misc/packed_bubbles.html
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, exemplars, img_list):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        
        offset_width = np.maximum(0, -10*int(np.ceil(np.min(self.bubbles[:, 0] - self.bubbles[:, 2]))))
                              
        offset_height = np.maximum(0, -10*int(np.ceil(np.min(self.bubbles[:, 1] - self.bubbles[:, 2]))))
        collage_width = 10*int(np.ceil(np.max(self.bubbles[:, 0] + self.bubbles[:, 2]))) + offset_width
        collage_height = 10*int(np.ceil(np.max(self.bubbles[:, 1] + self.bubbles[:, 2]))) + offset_height

        new_image = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
        
        for i in range(len(self.bubbles)):
            if self.bubbles[i, 2] > 1/20:
                im = Image.open(img_list[exemplars[i]])
                im = im.resize((int(20*self.bubbles[i, 2]), int(20*self.bubbles[i, 2])))
                bigsize = (im.size[0] * 3, im.size[1] * 3)
                mask = Image.new('L', bigsize, 0)
                draw = ImageDraw.Draw(mask) 
                draw.ellipse((0, 0) + bigsize, fill=255)
                mask = mask.resize(im.size, Image.ANTIALIAS)

                offset = (offset_width + 10*int(self.bubbles[i, 0] - self.bubbles[i, 2]), 
                          offset_height + 10*int(self.bubbles[i, 1] - self.bubbles[i, 2]))

                new_image.paste(im, offset, mask)
                         
        ax.imshow(new_image)

def plot_weights_over_time(weights, exemplars, info, num_cols=1, order=None):
    M = len(exemplars)
    T = len(weights)
    width_ratios = [num_cols*.5, num_cols*3]
    for i in range(1, num_cols):
        width_ratios.append(num_cols*.5)
        width_ratios.append(num_cols*3)
    xx = int(np.ceil(M/num_cols))
    yy = 2*num_cols
    fig, ax = plt.subplots(xx,yy, 
                           gridspec_kw={'width_ratios': [1, 2]*num_cols},
                          figsize=(yy*3, xx)
                          )

    x_vals = [1905, 1915, 1925, 1935, 1945, 1955, 1965, 1975, 1985, 1995, 2005, 2015]
    
    max_weight = np.max([np.max(weights[t]) for t in range(T)])
    
    col_ind = 0
    row_ind = 0
    img_list = info.filename
    if order is None:
        order = np.arange(M)
    for m in order:
        
        if row_ind >= np.ceil(M/num_cols):
            col_ind += 1
            row_ind = 0
        img = Image.open(img_list[exemplars[m]])
        ax[row_ind, 2*col_ind].imshow(img)
        ax[row_ind, 2*col_ind].set_axis_off()
        weights_over_time = [weights[t][m] for t in range(T)]
        
        ax[row_ind, 2*col_ind+1].plot(x_vals, weights_over_time)
        ax[row_ind, 2*col_ind+1].set_ylim(0, max_weight)
        ax[row_ind, 2*col_ind+1].set_xlim(1900, 2016)
        ax[row_ind, 2*col_ind+1].get_yaxis().set_visible(False)
        year = info.year[exemplars[m]]
        ax[row_ind, 2*col_ind+1].axvline(x=year, color='red')
        ax[row_ind, 2*col_ind+1].set(yticklabels=[])
        if row_ind<int(np.ceil(M/num_cols))-1:
            ax[row_ind, 2*col_ind+1].set(xticklabels=[])
        row_ind += 1
    plt.show()

def plot_ordered_weights(ordered_exemplars, ordered_weights, img_list, size=30, max_per_row=100, spacer=50):
    num_exemplars = len(ordered_exemplars)

    w, h = Image.open(img_list[0]).size

    collage_width = num_exemplars * w
    new_image = Image.new('RGB', (collage_width, h))
    cursor = (0,0)

    
    for k in range(num_exemplars):
        new_image.paste(Image.open(img_list[ordered_exemplars[k]]), cursor)
        cursor = (cursor[0]+w, 0)
    fig, axes = plt.subplots(1,1, figsize=(20,20))
    axes.imshow(new_image)
    plt.axis('off')
    plt.show()
