# Copyright (C) 2013 Wesley Baugh
import itertools
import multiprocessing
import Queue

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.animation import FuncAnimation

from infertweet.sentiment.constants import CHUNK_SIZE, TITLES, LABELS


def make_subplots():
    """Create the figure, axes, and lines."""

    def setup_axes(axes):
        """Setup look and feel of the axes."""
        for ax, title in zip(axes, TITLES):
            ax.set_title(title)
            ax.set_xlabel('training instances')
            ax.set_ylabel('performance')

            ax.set_xbound(0, 100)  # bound will change as needed.
            ax.set_ylim(0, 1)  # limit won't change automatically.

            # ax.xaxis.set_major_locator(MaxNLocator(10))
            # ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(MaxNLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax.grid(True)

            # Use current line labels to build legend.
            ax.legend(loc='upper center', ncol=len(LABELS))

    def make_lines(axes):
        """Create the lines for each axes."""
        lines = []
        marker = itertools.cycle('o^vds')
        for ax in axes:
            ax_lines = []
            for label in LABELS:
                x, y = [0], [0]
                line, = ax.plot(x, y, label=label)  # comma for unpacking.
                line.set_marker(next(marker))
                line.set_alpha(0.75)
                if label == 'Accuracy':
                    line.set_color('black')
                    line.set_zorder(0)  # Drawn first, so is underneath.
                ax_lines.append((line, x, y))
            lines.append(ax_lines)
        return lines

    def setup_figure(fig):
        """Setup look and feel of the figure."""
        fig.set_size_inches(12, 9, forward=True)
        fig.tight_layout()

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    lines = make_lines(axes)
    setup_axes(axes)
    setup_figure(fig)
    return fig, axes, lines


def get_data(queue):
    """Generate new data for the figure."""
    while 1:
        try:
            item = queue.get(timeout=0.05)
        except Queue.Empty:
            yield None
        except KeyboardInterrupt:
            return
        else:
            if item is None:
                return
            else:
                yield item


def update_figure(data, lines, axes):
    """Update the figure with new data."""
    if data is None:
        return []

    def update_lines(data, lines):
        """Append data to the appropriate lines."""
        updated = []
        for ax_lines in lines:
            for line, x, y in ax_lines:
                line_label = line.get_label()
                axes_label = line.get_axes().get_title()
                x.append(data[axes_label]['count'])
                y.append(data[axes_label][line_label])
                line.set_data(x, y)
                updated.append(line)
        return updated

    def update_axes(count, axes):
        """Rescale axes to fit the new data."""
        def round_nearest(x, base):
            return int(base * round(float(x) / base))

        for ax in axes:
            ax.set_xbound(0, round_nearest(count + CHUNK_SIZE, CHUNK_SIZE))
            # ax.relim()
            # ax.autoscale_view(True,True,True)

    count = max(data[x]['count'] for x in data)
    updated = update_lines(data, lines)
    update_axes(count, axes)

    return updated


def plot_worker(queue, animation=False):
    """Matplotlib worker."""

    def init():
        updated_lines = []
        for ax_lines in lines:
            for line, x, y in ax_lines:
                line.set_data([], [])
                updated_lines.append(line)
        return updated_lines

    fig, axes, lines = make_subplots()
    # Important to assign it to a variable, even if we don't use it.
    anim = FuncAnimation(fig=fig,
                         func=update_figure,
                         frames=lambda: get_data(queue),
                         fargs=(lines, axes),
                         interval=200,
                         repeat=False,
                         init_func=init,
                         blit=False)
    if animation:
        anim.save('plot.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


def normalized_confusion_matrix(confusion_matrix):
    """Normalize all values to be between 0 and 1."""
    norm_conf = []
    for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        if a == 0:
            return confusion_matrix
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    return norm_conf


def make_confusion():
    """Create the figure, axes, and lines."""

    def make_images(axes, empty_confusion):
        images = []
        annotations = []
        empty_confusion[0][0] = 1  # So that the colorbar will be correct.
        for ax in axes:
            confusion_image = ax.imshow(empty_confusion, cmap=plt.cm.jet,
                                        interpolation='nearest')
            images.append(confusion_image)
            ax.figure.colorbar(confusion_image, ax=ax, cmap=plt.cm.jet)
        return images, annotations

    def setup_axes(axes, empty_confusion):
        """Setup look and feel of the axes."""
        width = len(empty_confusion)
        height = len(empty_confusion[0])
        for ax, title in zip(axes, TITLES):
            ax.set_title(title)
            ax.set_aspect(1)
            alphabet = ['Neg', 'Neu', 'Pos']
            ax.set_xticks(range(width))
            ax.set_xticklabels(alphabet[:width])
            ax.set_yticks(range(height))
            ax.set_yticklabels(alphabet[:height])

    def setup_figure(fig):
        """Setup look and feel of the figure."""
        fig.set_size_inches(6, 9, forward=True)
        fig.tight_layout()

    fig, axes = plt.subplots(2, 1)
    empty_confusion = [[0] * 3] * 3
    images, annotations = make_images(axes, empty_confusion)

    setup_axes(axes, empty_confusion)
    setup_figure(fig)

    return fig, axes, images, annotations


def clear_annotations(annotations):
    for ax_annotation in annotations:
        for a in ax_annotation:
            a.remove()


def annotate_confusion_matrix(ax, confusion_matrix, annotations):
    width = len(confusion_matrix)
    height = len(confusion_matrix[0])

    annotations = []
    for x in xrange(width):
        for y in xrange(height):
            a = ax.annotate(str(confusion_matrix[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')
            annotations.append(a)
    return annotations


def update_confusion(data, images, axes, annotations):
    """Update the figure with new data."""
    if data is None:
        return []

    def update_images(data, images, axes, annotations):
        clear_annotations(annotations)
        annotations = []
        for ax, image, title in zip(axes, images, TITLES):
            confusion_matrix = data[title]['confusion']._confusion
            norm_conf = normalized_confusion_matrix(confusion_matrix)
            image.set_data(norm_conf)
            image.norm.vmin, image.norm.vmax = 0.0, 1.0
            annotations.append(annotate_confusion_matrix(ax, confusion_matrix,
                                                         annotations))
        return annotations

    annotations[:] = update_images(data, images, axes, annotations)

    return images


def confusion_worker(queue, animation=False):
    """Matplotlib worker."""

    def init():
        clear_annotations(annotations)
        for ax, image, title in zip(axes, images, TITLES):
            empty_confusion = [[0] * 3] * 3
            image.set_data(empty_confusion)
            # annotate_confusion_matrix(ax, empty_confusion, annotations)
        return images

    fig, axes, images, annotations = make_confusion()
    # Important to assign it to a variable, even if we don't use it.
    anim = FuncAnimation(fig=fig,
                         func=update_confusion,
                         frames=lambda: get_data(queue),
                         fargs=(images, axes, annotations),
                         interval=200,
                         repeat=False,
                         init_func=init,
                         blit=False)
    if animation:
        anim.save('confusion.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


def start_plot(plot_queue, confusion_queue, animation=False):
    """Uses multiprocessing to start matplotlib spawning GUIs."""
    if animation:
        raise NotImplementedError("Animation doesn't work with threading.")

    args = (plot_queue, animation)
    process = multiprocessing.Process(target=plot_worker, args=(args))
    # process.daemon = True
    process.start()

    args = (confusion_queue, animation)
    process = multiprocessing.Process(target=confusion_worker, args=(args))
    # process.daemon = True
    process.start()
