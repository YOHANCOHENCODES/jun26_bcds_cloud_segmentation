import matplotlib.pyplot as plt

from src.data.image_masks import ImageMask


def display_masked_image(image_data: ImageMask):
    '''Display given image with 3 visualizations: The image (left), the mask 
    (center) and the image with the mask as an overlay (right)'''
    # Create a figure with 3 sublots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Read the image
    img = plt.imread(image_data.path / image_data.filename)

    # Display the image
    axes[0].imshow(img)
    axes[0].set_title(image_data.filename)

    # Display the mask
    axes[1].imshow(image_data.mask_pixels, cmap='gray')
    axes[1].set_title(f'Mask: {image_data.filename}')

    # Display the image with a transparent overlay
    axes[2].imshow(img)
    axes[2].imshow(image_data.mask_pixels, cmap='Reds', alpha=0.4)
    axes[2].set_title(f'Mask region labeled as "{image_data.label_name}"')

    # Turn off the scales for all axes
    for ax in axes:
        ax.axis('off')
    plt.show()


# def create_segmap(image_id):
#     '''
#     Helper function to create a segmentation map for an image by image filename
#     '''
#     # open the image
#     image = np.asarray(Image.open(image_id))

#     # get masks for different classes
#     fish_mask = get_mask_by_image_id(image_id, 'Fish')
#     flower_mask = get_mask_by_image_id(image_id, 'Flower')
#     gravel_mask = get_mask_by_image_id(image_id, 'Gravel')
#     sugar_mask = get_mask_by_image_id(image_id, 'Sugar')

#     # label numpy map with 4 classes
#     segmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
#     segmap = np.where(fish_mask == 1, 1, segmap)
#     segmap = np.where(flower_mask == 1, 2, segmap)
#     segmap = np.where(gravel_mask == 1, 3, segmap)
#     segmap = np.where(sugar_mask == 1, 4, segmap)

#     # create a segmantation map
#     segmap = SegmentationMapOnImage(segmap, shape=image.shape, nb_classes=5)

#     return segmap


# def draw_labels(image, np_mask, label):
#     '''
#     Function to add labels to the image.
#     '''
#     if np.sum(np_mask) > 0:
#         x, y = 0, 0
#         x, y = np.argwhere(np_mask == 1)[0]

#         image = imgaug.imgaug.draw_text(
#             image, x, y, label, color=(255, 255, 255), size=50)
#     return image


# def draw_segmentation_maps(image_id):
#     '''
#     Helper function to draw segmantation maps and text.
#     '''
#     # open the image
#     image = np.asarray(Image.open(image_id))

#     # get masks for different classes
#     fish_mask = get_mask_by_image_id(image_id, 'Fish')
#     flower_mask = get_mask_by_image_id(image_id, 'Flower')
#     gravel_mask = get_mask_by_image_id(image_id, 'Gravel')
#     sugar_mask = get_mask_by_image_id(image_id, 'Sugar')

#     # label numpy map with 4 classes
#     segmap = create_segmap(image_id)

#     # draw the map on image
#     image = np.asarray(segmap.draw_on_image(np.asarray(image))
#                        ).reshape(np.asarray(image).shape)

#     image = draw_labels(image, fish_mask, 'Fish')
#     image = draw_labels(image, flower_mask, 'Flower')
#     image = draw_labels(image, gravel_mask, 'Gravel')
#     image = draw_labels(image, sugar_mask, 'Sugar')

#     return image

# # helper function to visualize several segmentation maps on a single image


# def visualize_several_maps(image_id):
#     '''
#     Function to visualize several segmentation maps.
#     INPUT:
#         image_id - filename of the image
#     '''
#     # open the image
#     image = np.asarray(Image.open(image_id))

#     # draw segmentation maps and labels on image
#     image = draw_segmentation_maps(image_id)

#     # visualize the image and map
#     side_by_side = np.hstack([
#         image
#     ])

#     labels = get_labels(image_id.split('/')[-1])

#     fig, ax = plt.subplots(figsize=(15, 7))
#     ax.axis('off')
#     plt.title('Segmentation maps:' + labels)
#     plt.legend()

#     ax.imshow(side_by_side)
