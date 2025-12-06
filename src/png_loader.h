#ifndef SIMPLE_PNG_LOADER_H
#define SIMPLE_PNG_LOADER_H

#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* load_png_rgb(const char* filename, int* width, int* height)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Erro: não foi possível abrir %s\n", filename);
        return NULL;
    }

    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        fclose(fp);
        fprintf(stderr, "Erro lendo cabeçalho de %s\n", filename);
        return NULL;
    }
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "Erro: arquivo %s não é PNG válido.\n", filename);
        fclose(fp);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr  = png_create_info_struct(png_ptr);
    if (!png_ptr || !info_ptr) {
        fprintf(stderr, "Erro criando estruturas PNG.\n");
        fclose(fp);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Erro durante leitura do PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    *width  = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    // Force 8-bit
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (color_type & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    if (!row_pointers) {
        fprintf(stderr, "Erro: sem memória (row_pointers).\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    unsigned char* tmpbuf = (unsigned char*)malloc((*height) * rowbytes);
    if (!tmpbuf) {
        fprintf(stderr, "Erro: sem memória (tmpbuf).\n");
        free(row_pointers);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < *height; i++)
        row_pointers[i] = tmpbuf + i * rowbytes;

    png_read_image(png_ptr, row_pointers);

    const size_t tight_rowbytes = (size_t)(*width) * 3u;
    unsigned char* outbuf = (unsigned char*)malloc((size_t)(*height) * tight_rowbytes);
    if (!outbuf) {
        fprintf(stderr, "Erro: sem memória (outbuf).\n");
        free(tmpbuf);
        free(row_pointers);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    if (rowbytes == tight_rowbytes) {
        memcpy(outbuf, tmpbuf, (size_t)(*height) * tight_rowbytes);
    } else {
        for (int y = 0; y < *height; ++y) {
            memcpy(outbuf + (size_t)y * tight_rowbytes, tmpbuf + (size_t)y * rowbytes, tight_rowbytes);
        }
    }

    // cleanup
    free(tmpbuf);
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return outbuf;
}

#endif // SIMPLE_PNG_LOADER_H
