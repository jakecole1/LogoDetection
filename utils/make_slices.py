import fitz
import os
import logging

logger = logging.getLogger(__name__)

# Define missing variables for backward compatibility
nestle_artwork_key_mapping = {}
nestle_fields_with_extra_padding = []

class MakeSlices:
    def __init__(self, client_name: str, stride: int = 250):
        self.client_name = client_name
        self.stride = stride
    

    def create_nxn_blocks(self, file_path: str, output_folder_path: str, block_size: int = 500, zoom_factor: float = 1.0):
        """
        Cut the PDF into 100x100 pixel blocks or smaller if there is excess.
        Each block will be saved as a separate image file.
        """
        doc = fitz.open(file_path)
        slice_metadata = {}
        page_num = 0
        os.makedirs(output_folder_path, exist_ok=True)

        page = doc.load_page(0)
        pw, ph = page.rect.width, page.rect.height
        
        # Calculate number of blocks needed
        blocks_x = int(pw // block_size) + (1 if pw % block_size > 0 else 0)
        blocks_y = int(ph // block_size) + (1 if ph % block_size > 0 else 0)
        
        logger.info(f"Creating {blocks_x}x{blocks_y} blocks of size {block_size}x{block_size} pixels")
        logger.info(f"Page dimensions: {pw}x{ph} pixels")
        
        block_count = 0
        
        for row in range(blocks_y):
            for col in range(blocks_x):
                x0 = col * block_size
                y0 = row * block_size
                x1 = min(x0 + block_size, pw)
                y1 = min(y0 + block_size, ph)
                
                clip_rect = fitz.Rect(x0, y0, x1, y1)
                
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                
                # Fix for pixmaps with alpha channel (transparency)
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Save the block image
                out_path = os.path.join(output_folder_path, f"block_{row}_{col}.png")
                pix.save(out_path)
                
                # Store metadata for this block
                slice_metadata[f"block_{row}_{col}"] = {
                    "page_num": page_num,
                    "pdf_bbox": {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1
                    },
                    "zoom_factor": zoom_factor,
                    "slice_wh": {
                        "width": clip_rect.width * zoom_factor,
                        "height": clip_rect.height * zoom_factor,
                    },
                    "page_wh": {
                        "width": pw,
                        "height": ph
                    },
                    "block_position": {
                        "row": row,
                        "col": col
                    }
                }
                
                block_count += 1
                logger.info(f"Created block {block_count}/{blocks_x * blocks_y}: {out_path}")
        
        doc.close()
        logger.info(f"Successfully created {block_count} blocks")
        return slice_metadata

    def make_slices_with_entities(self, file_path: str, bbox_from_full: dict, output_folder_path: str,
                                    entities_matched: list = [], pad: float=0.08, pad_more: float=0.15, zoom_factor: float=2.0):
            """
                Make slices for entities which are not matching to the source yet
                This saves cost and latency of parallel calls if comparison matches the entities before creating entity slices
            """
            doc = fitz.open(file_path)
            slice_metadata = {}
            page_num = 0
            os.makedirs(output_folder_path, exist_ok=True)

            page = doc.load_page(0)
            pw, ph = page.rect.width, page.rect.height

            for k in bbox_from_full.keys():
                if self.client_name == "NESTLE":
                    if k in nestle_artwork_key_mapping.keys():
                        mapped_key = nestle_artwork_key_mapping[k]
                    else:
                        mapped_key = k
                    fields_with_extra_padding = nestle_fields_with_extra_padding                
                elif self.client_name == "THE FERRERO GROUP":
                    mapped_key = k
                    fields_with_extra_padding = []
                else:
                    mapped_key = k
                    fields_with_extra_padding = []

                if mapped_key not in entities_matched:
                    logger.info(f"extracting entities for entity {k}")
                    for idx, bb in enumerate(bbox_from_full[k]["bbox"]):
                        x0, y0, x1, y1 = bb
                        
                        # create a slice with this bounding box
                        if k not in set(fields_with_extra_padding):
                            x0, y0, x1, y1 = x0-pad, y0-pad, x1+pad, y1+pad
                        else:
                            x0, y0, x1, y1 = x0-pad_more, y0-pad_more, x1+pad_more, y1+pad_more
                        # Ensure bounds are within page
                        x0 = max(0, x0)
                        y0 = max(0, y0)
                        x1 = min(1, x1)
                        y1 = min(1, y1)
                        
                        x0, y0, x1, y1 = x0 * pw,  y0 * ph, x1 * pw, y1 * ph
                        clip_rect = fitz.Rect(x0, y0, x1, y1)

                        # Render cropped area as a pixmap
                        mat = fitz.Matrix(zoom_factor, zoom_factor)  # Zoom in
                        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                        
                        # Fix for pixmaps with alpha channel (transparency)
                        if pix.n > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        # Create zoomed slice image
                        out_path = os.path.join(output_folder_path, f"{k}@{idx}.png")
                        pix.save(out_path)
                        # scaled coordinates in the slices
                        slice_metadata[f"{k}@{idx}"] = {
                                "page_num": page_num,
                                "pdf_bbox": {
                                    "x0": x0,
                                    "y0": y0,
                                    "x1": x1,
                                    "y1": y1
                                },
                                "zoom_factor": zoom_factor,
                                "slice_wh": {
                                    "width": clip_rect.width * zoom_factor,  # width_zoomed,
                                    "height": clip_rect.height * zoom_factor,  # height_zoomed
                                },
                                "page_wh": {
                                    "width": pw,
                                    "height": ph
                                }
                            }
            doc.close()
            return slice_metadata



if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create 100x100 blocks from the PDF
    pdf_file = "SKUR-AWK-2502-00249_8005164-1_Coca-Cola_Coca-Cola Zero Caffeine free_CCZCF_Can Slim_200ml_25-0042-FR_FRA_samplingminica_Ardagh_CCEP.pdf"
    output_folder = "output"
    
    # Create MakeSlices instance and generate blocks
    slicer = MakeSlices(client_name="")
    metadata = slicer.create_nxn_blocks(
        file_path=pdf_file,
        output_folder_path=output_folder,
        block_size=500,
        zoom_factor=2.0
    )
    
    print(f"Generated {len(metadata)} blocks")
    print("Block metadata:")
    for block_name, block_info in metadata.items():
        print(f"  {block_name}: {block_info['pdf_bbox']} (size: {block_info['slice_wh']['width']}x{block_info['slice_wh']['height']})")