"""
Plot a frame UMAP projection in latent space with its corresponding
reconstructed or original (i.e., input) frames appearing as tooltips.

Adapted from jack's tooltip_plot.py in AVA repo.
"""

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
import umap

def tooltip_plot_DC(dataset, embeddings, recons, output_dir='', embedding_type='latent_mean_umap', \
type = 'orig', img_range=[0, 10], title="", n=3100, grid=False, img_format='.jpg'):
	"""
	Mouse VAE version of `tooltip_plot`.
	Parameters
	----------
    dataset: DataSet object w/ original frames (unshuffled)
    embeddings: UMAP LS Mean embeddings (created using project_latent method.)
	embedding_type : str, optional
		Defaults to ``'latent_mean_umap'``.
	output_dir : str
		Directory where html and jpegs are written.
	num_imgs : int, optional
		Number of points with tooltip images. Defaults to ``2000``.
	title : str, optional
		Title of plot. Defaults to ``''``.
	n : int, optional
		Total number of scatterpoints to plot. Defaults to ``80000``.
	grid : bool, optional
		Show x and y grid? Defaults to ``False``.
	img_format : str, optional
		Filetype for tooltip spectrograms. Defaults to ``'.jpg'``.
	"""
	embedding = embeddings
	images = dataset
	if type == 'recons':
		images = recons
	output_dir = os.path.join(output_dir, "trajectory_tooltip_plot")
	print("writing tooltip plot to", output_dir)
	tooltip_plot(embedding, images, output_dir=output_dir, \
		title=title, n=n, grid=grid, type=type, img_range=img_range)


def tooltip_plot(embedding, images, output_dir='temp', title="",
	n=3100, grid=False, type='orig', img_range=[0, 10]):
	"""
	Create a scatterplot of the embedding with spectrogram tooltips.
	Parameters
	----------
	embedding : numpy.ndarray
		The scatterplot coordinates. Shape: (num_points, 2)
	images : numpy.ndarray
		2D images (frames) corresponding to the scatterpoints. Shape:
		(num_points, height, width)
	output_dir : str, optional
		Directory where html and jpegs are written. Deafaults to "temp".
	title : str, optional
		Title of plot. Defaults to ''.
	n : int, optional
		Total number of scatterpoints to plot. Defaults to 3100.
	grid : bool, optional
		Show x and y grid? Defaults to `False`.
	"""
	if type == 'orig':
		perm_imgs = _get_orig_images(img_range, images)
	else:
		perm_imgs = images[:, :, img_range[0]:img_range[1]]
		#perm_imgs = recons
	n = min(len(embedding), n)
	num_imgs = img_range[1] - img_range[0]
	_write_images(embedding, perm_imgs, output_dir=output_dir, num_imgs=num_imgs, n=n)
	output_file(os.path.join(output_dir, "main.html"))
	source = ColumnDataSource(
			data=dict(
				x=embedding[img_range[0]:img_range[1],0],
				y=embedding[img_range[0]:img_range[1],1],
				imgs = [os.path.join(output_dir, str(i)+'.jpg') for i in range(num_imgs)],
			)
		)
	source2 = ColumnDataSource(
			data=dict(
				x=embedding[:,0],
				y=embedding[:,1],
			)
		)
	source3 = ColumnDataSource(
			data=dict(
				x=embedding[img_range[0]:img_range[1],0],
				y=embedding[img_range[0]:img_range[1],1],
			)
		)
	p = figure(plot_width=800, plot_height=600, title=title)
	p.scatter('x', 'y', size=3, fill_color='blue', fill_alpha=0.1, source=source2)
	p.line('x', 'y', line_color='green', line_width=2, source=source3)
	tooltip_points = p.scatter('x', 'y', size=5, fill_color='red', source=source)
	hover = HoverTool(
			renderers=[tooltip_points],
			tooltips="""
			<div>
				<div>
					<img
						src="@imgs" height="160" alt="@imgs" width="120"
						style="float: left; margin: 0px 0px 0px 0px;"
						border="1"
					></img>
				</div>
			</div>
			"""
		)
	p.add_tools(hover)
	p.title.align = "center"
	p.title.text_font_size = "25px"
	p.axis.visible = grid
	p.xgrid.visible = grid
	p.ygrid.visible = grid
	show(p)

def _get_orig_images(img_range, images):
	perm_imgs = []
	for  i in range(img_range[0], img_range[1]):
		image = images.__getitem__(i)
		frame = image['frame'].numpy()
		perm_imgs.append(frame)
	res = np.array(perm_imgs)
	return res

def _save_image(data, filename):
	"""https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/"""
	sizes = np.shape(data)
	print(sizes)
	height = float(sizes[1])
	width = float(sizes[2])
	fig = plt.figure()
	fig.set_size_inches(width/height, 1, forward=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(data.transpose(1,2,0), cmap='gray')
	plt.imshow(fig)
	plt.savefig(filename, dpi=height)
	plt.close('all')


def _write_images(embedding, images, output_dir='/temp', num_imgs=10, n=3100):
	print(os.getcwd())
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	X = embedding[:,0]
	Y = embedding[:,1]
	for i in range(num_imgs):
		try:
			print(images[i].shape)
			images[i] = images[i].swapaxes(0,3)
			_save_image(images[i][:, :, :, :], os.path.join(output_dir, str(i) + '.jpg'))
			print(output_dir, str(i))
		except(ValueError):
			pass
	return embedding



if __name__ == '__main__':
	pass
