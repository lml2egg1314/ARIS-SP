
% addpath('JPEG_Toolbox')
addpath('jpegtbx');

file_num = 10000;
% QF = 60;
QF = params.quality;

listNum = 1;
payload = params.payload;

cover_dir = params.cover_dir;

stego_dir = params.stego_dir;

% extract_and_classify(QF, cover_dir, ori_stego_dir, params, payload);
extract_and_classify(QF, cover_dir, stego_dir, params, payload);

function extract_and_classify(QF, cover_dir, stego_dir, params, payload)
    
    total_start = tic;
    listNum = params.listNum;
    dataset_division_array = load('BossBase_index');
    dataset_division = dataset_division_array.index;
    
%     dataset_division = randperm(10000);
    training_set = dataset_division(1:5000);
    test_set = dataset_division(5001:end);
    
    
% cover_dir = '.';

    file_list = dir([cover_dir, '/*.jpg']);
    
    
    cover_set = {};
    stego_set = {};

    for index = 1 : 10000
%         file_index = training_set(index);
%         file_index = index;
        file_name = sprintf('%d.jpg',index);
%         file_name = file_list(file_index).name;
        cover_set{end + 1} = sprintf('%s/%s', cover_dir, file_name);
        stego_set{end + 1} = sprintf('%s/%s', stego_dir, file_name);
    end
%     for index = 1 : 5000
%         file_index = test_set(index);
% %         file_index = index;
%         file_name = file_list(file_index).name;
%         cover_set{end + 1} = sprintf('%s/%s', cover_dir, file_name);
%         stego_set{end + 1} = sprintf('%s/%s', stego_dir, file_name);
%     end
   
    cover_features_path = sprintf('/data1/lml/watermarking/dctr_cover_features_ln%d_QF%d.mat',listNum, QF);
    if params.start == 0
        fprintf('Extract GFR features ---- \n');
        cover_features = extract_dctr(cover_set, QF);
        save(cover_features_path, 'cover_features');
    else
        cover_features_mat = load(cover_features_path);
        cover_features = cover_features_mat.cover_features;
    end
        
     
        fprintf('Extract  GFR features ----- \n');
        stego_features = extract_dctr(stego_set, QF);

        test_acc = ensemble_classify(cover_features, stego_features, training_set, test_set);

        total_end = toc(total_start);

        fprintf('GFR and ensemble results ----- \n')
        fprintf('Test accuracy for # %s-%.2f-jpeg-Q%d #: %.4f \n', 'j_uniward', payload,  QF, test_acc);

        file_id = fopen('acc_log_dmmr_parameter.txt','a');
        fprintf(file_id, '%s \n', stego_dir);
        fprintf(file_id,'dctr_%s-%.2f-jpeg-Q%d: %.4f\n', 'j_uniward',  payload, QF, test_acc);
        fclose(file_id);

        fprintf('Total time: %.2f seconds. \n', total_end);
        fprintf('------------------------- \n')
    end



function dctr_features = extract_dctr(image_set, QF)
    extract_start = tic;
    file_num = length(image_set);
    dctr_features = zeros(file_num, 8000);
 
    parfor i = 1:file_num
        image_item = image_set{i};
%         jpeg_path = [jpeg_dir, num2str(i), '.jpg'];
        j_struct = jpeg_read(image_item);
       
        j_f = DCTR(j_struct, QF);
%         j_f = GFR(image_item, QF);
        
        dctr_features(i, :) = j_f;
        
    end
    extract_end = toc(extract_start);

    fprintf('DCTR extracted %d cover images in %.2f seconds, in average %.2f seconds per image. \n\n', file_num, extract_end, extract_end / file_num);


end


function [test_acc] = ensemble_classify(cover_features, stego_features, training_set, testing_set)

  train_cover = cover_features(1:5000, :);
  train_stego = stego_features(1:5000, :);

  test_cover = cover_features(5001:10000, :);
  test_stego = stego_features(5001:10000, :);

  settings = struct('verbose', 2);

  train_start = tic;

  fprintf('Ensemble train start ----- \n');

  [trained_ensemble,results] = ensemble_training(train_cover, train_stego, settings);

  train_end = toc(train_start);


  fprintf('\n');

  test_start = tic;

  fprintf('Ensemble test start ----- \n');

  test_results_cover = ensemble_testing(test_cover, trained_ensemble);
  test_results_stego = ensemble_testing(test_stego, trained_ensemble);

  test_end = toc(test_start);


  % Predictions: -1 stands for cover, +1 for stego
  false_alarms = sum(test_results_cover.predictions ~= -1);
  missed_detections = sum(test_results_stego.predictions ~= +1);

  num_testing_samples = size(test_cover, 1) + size(test_stego, 1);

  testing_error = (false_alarms + missed_detections) / num_testing_samples;

  fprintf('Train time: %.2f seconds, Test time: %.2f seconds. \n\n', train_end, test_end);

  test_acc = 1 - testing_error;

end

