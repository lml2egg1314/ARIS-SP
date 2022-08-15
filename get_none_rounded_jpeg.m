
addpath('ddmr/jpegtbx');

QFs = [65, 75];
image_size = 256;

for i = 1:2
    
    quality = QFs(i);
    cover_dir = sprintf( '/data1/lml/watermarking/BB-cover-resample-%d-jpeg-%d',image_size, quality)
    % get_nr(cover_dir);
    for j = 1:5
        payload = 0.02*j;
        stego_dir = sprintf('%s-dmmr-upward-payload-%.2f-use-dcts-13',cover_dir, payload) 
        if i==1 && j == 5
            continue;
        end
        get_nr(stego_dir);    
    end
end




function get_nr(input_jpeg_dir)

output_jpeg_dir = sprintf('%s-none-round', input_jpeg_dir);
% output_jpeg_dir = sprintf('%s-dct', input_jpeg_dir)
if ~exist(output_jpeg_dir, 'dir')
    mkdir(output_jpeg_dir);
end

parfor index = 1 : 20000
% for index = 5818 : 5818

  input_jpeg_path = [input_jpeg_dir, '/', num2str(index), '.jpg'];
  output_jpeg_path = [output_jpeg_dir,'/', num2str(index), '.mat'];

  c_struct = jpeg_read(input_jpeg_path);

  [output_image, ~] = dct2spatial(c_struct);

  img = output_image;


  save_dct(output_jpeg_path, img);

end
end
function save_dct(output_path, img)

save(output_path, 'img', '-v6');
end

function [s_spatial, s_coef] = dct2spatial(s_struct)

  s_coef = s_struct.coef_arrays{1};
  s_quant = s_struct.quant_tables{1};

  dequntized_s_coef = dequantize(s_coef, s_quant);
  s_spatial = ibdct(dequntized_s_coef, 8) + 128;

end
