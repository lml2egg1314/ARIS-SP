use_dcts = [7,15];
for i = 2
    for j = 5
        payload = 0.02*j;
        QF = 55+10*i;
        image_size = 256;
%         if i == 1 && j ==1
%             base_dir = '/data1/lml/watermarking';
%             image_set = 'BossBase-1.01';
%             cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
%           
%             stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
%         else
%             myp2s = [0.1, 0.01, 0.001,0.0001, 0.005, 0.01, 0.05, 0.1];
            myp2 = 0.5;
            myp1s = [0.99, 0.95,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
%             for kk = 1:5
%                 myp1 = 1-0.1*kk;
%                 myp2 = 1 - myp2s(kk);
            tic;
%             [cover_dir, stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload);
              [cover_dir, stego_dir] = Generate_ad(QF, image_size, payload, myp1s, myp2);
%                 base_dir = '/data1/lml/watermarking';
%                 image_set = 'BossBase-1.01';
%                 cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
%     
%                 stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
            toc;
%         end
            params.payload = payload;
            params.quality = QF;
            params.cover_dir = cover_dir;
            ori_stego_dir = stego_dir;
            for k = 5
                myp1 = myp1s(k);
                params.stego_dir = sprintf('%s-ae1-%.2f',ori_stego_dir, myp1);
    %             params.stego_dir = stego_dir;
                params.start = 1;
                params.listNum = 1;
                params.myp1 = myp1;
                params.myp2 = myp2;
                disp(params);
                dctr_and_ensemble_from_image1;
            
            end
    end
end

function [cover_dir,stego_dir] = Generate_ad(QF, image_size, payload, myp1s, myp2)
%% ����ѹ���޸ĵĲ��Դ���
%     clear all;
%     clc;
%%  ��������
    cover_QF = QF;
    attack_QF = QF;
    addpath(fullfile('jpegtbx'));
    addpath(fullfile('stc_matlab'));
    base_dir = '/data1/lml/watermarking';
    image_set = 'BB';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
    afterchannel_stego_dir = sprintf('%s/DMMR/re_stego', base_dir); 
    ori_stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
    stego_dir = sprintf('%s-dmmr-new-payload-%.2f', cover_dir, payload); 
    mkdir(stego_dir);
    
    msg_dir = sprintf('%s-dmmr-payload-%.2f-msg', cover_dir, payload);
    cost_dir = sprintf('%s-dmmr-cost', cover_dir);
    
    for p1_ind = 5
        myp1 = myp1s(p1_ind);
        ae1_stego_dir = sprintf('%s-ae1-%.2f', stego_dir, myp1);
        ae1_stego_recom_dir = sprintf('%s-recom', ae1_stego_dir);
        if ~exist(ae1_stego_dir,'dir')
            mkdir(ae1_stego_dir); 
        end  
        if ~exist(ae1_stego_recom_dir,'dir')
            mkdir(ae1_stego_recom_dir); 
        end 
    end
    ae2_stego_dir = sprintf('%s-ae2', stego_dir);
    if ~exist(ae2_stego_dir,'dir')
        mkdir(ae2_stego_dir); 
    end  
%     ae3_stego_dir = sprintf('%s-ae3', stego_dir);
%     if ~exist(ae3_stego_dir,'dir')
%         mkdir(ae3_stego_dir); 
%     end  
    grad_dir = sprintf('%s-covnet-grad-1', ori_stego_dir);
    
    cover_num = 20000; 
%     bit_error_rate = zeros(length(myp1s),cover_num); 
    bit_error_rate = zeros(1,cover_num); 
 
    
    %%
    emb_fre = ones(8);
    for i  =1:8
        for j = 1:8
            if i+j == 5 || i+j == 6
                emb_fre(i,j) = 0;
            end
        end
    end
    rep_emb_fre = repmat(emb_fre, image_size/8, image_size/8);
    rep_emb_fre = logical(rep_emb_fre);
%%  ��ϢǶ��    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        
        stego_path = fullfile([stego_dir,'/',index_str,'.jpg']); 
        cost_path = fullfile([cost_dir,'/',num2str(i_img),'.mat']);
        afterchannel_stego_Path = fullfile([afterchannel_stego_dir,'/',index_str,'.jpg']);  
        ae_stego_recom_path = sprintf('%s/%s.jpg', ae2_stego_dir, index_str);
        
        grad_path = fullfile([grad_dir, '/', index_str, '.mat']);
        msg_path = fullfile([msg_dir,'/',num2str(i_img),'.mat']);
        
        
        
        %%
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};  
        C_QUANT = C_STRUCT.quant_tables{1}; %����ͼ��������
%         nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
%  ��������ȷֲ��Ķ�����ԭʼ�������?
        
        [raw_msg,~] = load_msg(msg_path);
        raw_msg_len = length(raw_msg);
        n = 255; k = 251; m =8 ;    % RS�������??? 
        [rho1_P,rho1_M] = load_cost(cost_path);
%         [rho1_P,rho1_M] = J_UNIWARD_D(cover_path,1);%ʹ��J_UNIWARD��д�㷨����DCTϵ���Ƕ����???
        [cover_lsb, change, rho] = DMMR(cover_path, rho1_P, rho1_M, C_QUANT);%����ͨ����Ƽ��������У��޸ľ��룬�޸Ĵ��
        %  ����STC������ϢǶ��  
        [stc_msg,stc_c1,stc_c2, code] = stc_embedding(raw_msg, cover_path, cover_lsb, rho, change,cover_QF, stego_path,n,k,m,ae_stego_recom_path,C_QUANT,attack_QF);
%%  ģ���ŵ�ѹ�� 
        
        
%         ae2_stego_path = fullfile([ae2_stego_dir,'/',index_str,'.jpg']);  
%         ae3_stego_path = fullfile([ae3_stego_dir,'/',index_str,'.jpg']);  
        grad_mat = load(grad_path);
        grad = grad_mat.cover_grad;
        pred = grad_mat.pred;
        cover_grad = squeeze(grad(1,:,:));
        stego_grad = squeeze(grad(2,:,:));
        grad_same = (sign(cover_grad) == sign(stego_grad));
        
%         if pred(2) == 0
%             copyfile(stego_path, ae1_stego_path);
%             copyfile(stego_path, ae2_stego_path);
%             copyfile(stego_path, ae3_stego_path);
%             continue;
%         end
        
%         mod_stego_grad = stego_grad .* rep_emb_fre;
%         mod_cover_grad = cover_grad .* ~rep_emb_fre;
        

%         nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
        
        S_STRUCT = jpeg_read(stego_path);
        S_COEFFS = S_STRUCT.coef_arrays{1};  
%         S_QUANT = S_STRUCT.quant_tables{1}; 
        ae_S_STRUCT = C_STRUCT;
        
        Modifications = S_COEFFS - C_COEFFS;
%         max(max(Modifications))
%         min(min(Modifications))
%         sum(sum(abs(Modifications)))
%         sum(sum(abs(Modifications .* ~rep_emb_fre)))
%         pause;
        %% Adversarial Embedding For the elements among robust embedding domain
%         myp1 = 0.01;
        temp_cover_grad = cover_grad;
        flat_cover_grad = reshape(temp_cover_grad,1,image_size*image_size);
        [x_grad, ~] = sort(abs(flat_cover_grad));
        
        for p1_index = 5
            myp1 = 0.7;
            ae1_stego_path = sprintf('%s-ae1-%.2f/%s.jpg', stego_dir, myp1, index_str);
            
            ae1_stego_recom_path = sprintf('%s-ae1-%.2f-recom/%s.jpg', stego_dir, myp1, index_str);
            
            % greater p grad
            high_cover_grad = x_grad(max(1,round(image_size*image_size*myp1)));
    %         pause;
            abs_cover_grad = (abs(temp_cover_grad)>high_cover_grad);
            ae_cover_mod_locations = logical(~rep_emb_fre .* abs_cover_grad .* abs(Modifications) .* grad_same);
    %         sum(sum(ae_cover_mod_locations))

            ae_cover_mod_signs = -sign(cover_grad);
            Modifications(ae_cover_mod_locations) = ae_cover_mod_signs(ae_cover_mod_locations);

            ae1_COEFFS = C_COEFFS + Modifications;
            ae_S_STRUCT.coef_arrays{1} = ae1_COEFFS;
            jpeg_write(ae_S_STRUCT, ae1_stego_path);
            recompression(ae1_stego_path,QF,ae1_stego_recom_path,C_QUANT,QF,code)


    %         [msg, stc_msg, stc_c1, stc_c2] = load_msg(msg_path);
            imwrite(imread(ae1_stego_path),afterchannel_stego_Path,'quality',attack_QF);    

    %%  ��Ϣ��ȡ
            [stc_decoded_msg] = stc_extracting(afterchannel_stego_Path, stc_msg,stc_c1, C_QUANT,n,k,m,stc_c2);   

    %%  ����ÿ��ͼ���������?????      
            bit_error = double(raw_msg) - double(stc_decoded_msg);
            bit_error_number = sum(abs(bit_error));
            bit_error_rate(1,i_img) = bit_error_number/raw_msg_len;
        end
        
        
%         temp_pre_rhoP1 = pre_rhoP1;
%         temp_pre_rhoM1 = pre_rhoM1;
% 
%         flat_pre_rhoP1 = reshape(temp_pre_rhoP1,1,image_size*image_size);
%         [x_P1, ~] = sort(flat_pre_rhoP1);
%         flat_pre_rhoM1 = reshape(temp_pre_rhoM1,1,image_size*image_size);
%         [x_M1, ~] = sort(flat_pre_rhoM1);
%         
    end
    
    ave_error_rate = mean(bit_error_rate,2);
    save_pp = sprintf('bit_error_rate_payload_%.2f.mat',payload); 
    save(save_pp,'ave_error_rate');
    file_id = fopen('ddmr_log_adv.txt', 'a');
    
%     fprintf(file_id,'image_size-%d-QF-%d-%s-p1-%.2f\n',image_size, QF, ['payload: ',num2str(payload),'  ave_error_rate: ',num2str(ave_error_rate)], myp1s);  
    fprintf(file_id,'Image_size: %d; QF: %d; Payload: %.2f;',image_size, QF, payload);  
%     fprintf(file_id,'      Attack_QF: %.8f; BER: %.8f \n',[myp1s; ave_error_rate']);  
    fprintf(file_id,'      Advesarial magnitude: %.8f; BER: %.8f \n',0.70, ave_error_rate);  
    fclose(file_id);
end
function save_msg(msg_path, msg, stc_msg, stc_c1, stc_c2)
save(msg_path,'msg', 'stc_msg', 'stc_c1', 'stc_c2');
end

function [rho1_P, rho1_M] = load_cost(cost_path)
cost_mat = load(cost_path);
rho1_P = cost_mat.rho1_P;
rho1_M = cost_mat.rho1_M;
end


function [msg, stc_msg, stc_c1, stc_c2] = load_msg(msg_path)
msg_mat = load(msg_path);
msg = msg_mat.msg;
stc_msg = msg_mat.stc_msg;
stc_c1 = msg_mat.stc_c1;
stc_c2 = msg_mat.stc_c2;

end

function [cover_dir,stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload)
%% ����ѹ���޸ĵĲ��Դ���
%     clear all;
%     clc;
%%  ��������
    addpath(fullfile('jpegtbx'));
    addpath(fullfile('stc_matlab'));
    base_dir = '/data1/lml/watermarking';
    image_set = 'BossBase-1.01';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
    recompress_dir=sprintf('%s/DMMR/recompress', base_dir); 
    if ~exist(recompress_dir,'dir')
        mkdir(recompress_dir); 
    end 
    stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
    if ~exist(stego_dir,'dir')
        mkdir(stego_dir); 
    end  
    afterchannel_stego_dir = sprintf('%s/DMMR/re_stego', base_dir); 
    if ~exist(afterchannel_stego_dir,'dir')
        mkdir(afterchannel_stego_dir); 
    end  
    cover_num = 10000; 
    cover_QF = QF; 
    attack_QF = QF; 
%     payload = 0.1; 
    bit_error_rate = zeros(1,cover_num); 
%%  ��ϢǶ��    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        recompress_Path=fullfile([ recompress_dir,'/',index_str,'.jpg']);
        stego_path = fullfile([stego_dir,'/',index_str,'.jpg']);    
        afterchannel_stego_Path = fullfile([afterchannel_stego_dir,'/',index_str,'.jpg']);   
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};  
        C_QUANT = C_STRUCT.quant_tables{1}; %����ͼ��������
        nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
%  ��������ȷֲ��Ķ�����ԭʼ�������?
        raw_msg_len = ceil(payload*nzAC);
        if raw_msg_len<=1 %����BOSSbase��ݼ��д��ڲ��ʺ�Ƕ��Ĳ�����Ƕ��
            imwrite(imread(cover_path),recompress_Path,'quality',cover_QF);
            imwrite(imread(recompress_Path),stego_path,'quality',cover_QF);
            imwrite(imread(stego_path),afterchannel_stego_Path,'quality',attack_QF);
            bit_error_rate(1,i_img)=0;
            continue;
        end
        raw_msg = round( rand(1,raw_msg_len) ); %ԭʼ������Ϣ��������    
        n = 255; k = 251; m =8 ;    % RS�������????? 
        [rho1_P,rho1_M] = J_UNIWARD_D(cover_path,1);%ʹ��J_UNIWARD��д�㷨����DCTϵ���Ƕ����?????
        [cover_lsb, change, rho] = DMMR(cover_path, rho1_P, rho1_M, C_QUANT);%����ͨ����Ƽ��������У��޸ľ��룬�޸Ĵ��
        %  ����STC������ϢǶ��  
        [stc_n_msg_bits,stc_n_msg_bits_check,l_check_check] = stc_embedding(raw_msg, cover_path, cover_lsb, rho, change,cover_QF, stego_path,n,k,m,recompress_Path,C_QUANT,attack_QF);
%%  ģ���ŵ�ѹ�� 
        imwrite(imread(stego_path),afterchannel_stego_Path,'quality',attack_QF);    
      
%%  ��Ϣ��ȡ
        [stc_decoded_msg] = stc_extracting(afterchannel_stego_Path, stc_n_msg_bits,stc_n_msg_bits_check, C_QUANT,n,k,m,l_check_check);   

%%  ����ÿ��ͼ���������?????      
        bit_error = double(raw_msg) - double(stc_decoded_msg);
        bit_error_number = sum(abs(bit_error));
        bit_error_rate(1,i_img) = bit_error_number/raw_msg_len;
        
%% ����ÿ�ų�����
%  ���ÿ��ͼ���������               
%         fprintf('%s\n',['payload: ',num2str(payload),'  image_number: ',index_str,'  error_rate: ',num2str(bit_error_rate(1,i_img))]);  %
    end
%  �������ͼ���ƽ��������
    ave_error_rate = mean(bit_error_rate);
    file_id = fopen('ddmr_log.txt', 'a');
    
    fprintf(file_id,'image_size-%d-QF-%d-%s\n',image_size, QF, ['payload: ',num2str(payload),'  ave_error_rate: ',num2str(ave_error_rate)]);  
    fclose(file_id);
end
