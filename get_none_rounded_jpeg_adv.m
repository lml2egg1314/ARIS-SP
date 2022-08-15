
addpath('ddmr/jpegtbx');

QFs = [65, 75];
image_size = 256;




for i = 1:2
    for j = 1:5
        % if i == 1 && j == 5
        %     continue;
        % end
    payload = 0.02 * j;
    quality = QFs(i);
    cover_dir = sprintf( '/data1/lml/watermarking/BB-cover-resample-%d-jpeg-%d',image_size, quality);

    stego_dir = sprintf('%s-dmmr-upward-new-payload-%.2f-use-dcts-13', cover_dir, payload); 
    myp1s = [0.99, 0.95,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
%     get_nr(cover_dir);
    for p1_ind = 5
        myp1 = myp1s(p1_ind);
        ae1_stego_dir = sprintf('%s-ae1-%.2f', stego_dir, myp1)
        get_nr(ae1_stego_dir);
%         ae1_stego_recom_dir = sprintf('%s-recom', ae1_stego_dir);
%     for j = 1:2
%         payload = 0.05*j;
%         stego_dir = sprintf('%s-dmmr-payload-%.2f',cover_dir, payload)     
%         get_nr(ae1_stego_dir);    
    end
    end
end




function get_nr(input_jpeg_dir)

output_jpeg_dir = sprintf('%s-none-round', input_jpeg_dir)
% output_jpeg_dir = sprintf('%s-dct', input_jpeg_dir);
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
