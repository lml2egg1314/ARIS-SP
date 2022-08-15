use_dcts = [7,15];
warning off;
for i = 1:2
    params.start = 1;
    for j = 1:9
        payload = 0.01*j;
        if j == 5 
            continue;
        end
        QF = 55+10*i;
        image_size = 256;
        
%         if i == 1 && j ==1
%             base_dir = '/data1/lml/watermarking';
%             image_set = 'BossBase-1.01';
%             cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
%           
%             stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
%         else
            tic;
            [cover_dir, stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload, params);
            toc;
%         end
        params.payload = payload;
        params.quality = QF;
        params.cover_dir = cover_dir;
        params.stego_dir = stego_dir;
        
        params.listNum = 1;
        disp(params);
        dctr_and_ensemble_from_image1;
        params.start = 1;
    end
end

function [cover_dir,stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload, params)
%% ����ѹ���޸ĵĲ��Դ���
%     clear all;
%     clc;
%%  ��������
    addpath(fullfile('jpegtbx'));
    addpath(fullfile('stc_matlab'));
    para_start = params.start;
    base_dir = '/data1/lml/watermarking';
%     image_set = 'BossBase-1.01';
    image_set = 'BB';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
    recompress_dir=sprintf('%s/DMMR/recompress', base_dir); 
    if ~exist(recompress_dir,'dir')
        mkdir(recompress_dir); 
    end
    cost_dir = sprintf('%s-dmmr-cost', cover_dir); 
    if ~exist(cost_dir,'dir')
        mkdir(cost_dir); 
    end  
    stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
    if ~exist(stego_dir,'dir')
        mkdir(stego_dir); 
    end  
    msg_dir = sprintf('%s-dmmr-payload-%.2f-msg', cover_dir, payload); 
    if ~exist(msg_dir,'dir')
        mkdir(msg_dir); 
    end  
    afterchannel_stego_dir = sprintf('%s/DMMR/re_stego', base_dir); 
    if ~exist(afterchannel_stego_dir,'dir')
        mkdir(afterchannel_stego_dir); 
    end  
    cover_num = 20000; 
    cover_QF = QF; 
    attack_QF = QF; 
%     payload = 0.1; 
    bit_error_rate = zeros(1,cover_num); 
%%  ��ϢǶ��    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        recompress_Path=fullfile([ recompress_dir,'/',index_str,'.jpg']);
        stego_Path = fullfile([stego_dir,'/',index_str,'.jpg']);   
        msg_Path = fullfile([msg_dir,'/',num2str(i_img),'.mat']);
        cost_path = fullfile([cost_dir,'/',num2str(i_img),'.mat']);
        afterchannel_stego_Path = fullfile([afterchannel_stego_dir,'/',index_str,'.jpg']);   
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};  
        C_QUANT = C_STRUCT.quant_tables{1}; %����ͼ��������
        nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
%  ��������ȷֲ��Ķ�����ԭʼ�������?
        raw_msg_len = ceil(payload*nzAC);
        if raw_msg_len<=1 %����BOSSbase��ݼ��д��ڲ��ʺ�Ƕ��Ĳ�����Ƕ��
            imwrite(imread(cover_path),recompress_Path,'quality',cover_QF);
            imwrite(imread(recompress_Path),stego_Path,'quality',cover_QF);
            imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);
            bit_error_rate(1,i_img)=0;
            continue;
        end
        raw_msg = round( rand(1,raw_msg_len) ); %ԭʼ������Ϣ��������
        
        n = 255; k = 251; m =8 ;    % RS�������?? 
       
        if para_start == 0
            [rho1_P,rho1_M] = J_UNIWARD_D(cover_path,1);
            save_cost(cost_path, rho1_P, rho1_M);
        else
            [rho1_P,rho1_M] = load_cost(cost_path);
        end
        [cover_lsb, change, rho] = DMMR(cover_path, rho1_P, rho1_M, C_QUANT);%����ͨ����Ƽ��������У��޸ľ��룬�޸Ĵ��
        %  ����STC������ϢǶ��  
        [stc_n_msg_bits,stc_n_msg_bits_check,l_check_check] = stc_embedding(raw_msg, cover_path, cover_lsb, rho, change,cover_QF, stego_Path,n,k,m,recompress_Path,C_QUANT,attack_QF);
%%  ģ���ŵ�ѹ�� 
        save_msg(msg_Path, raw_msg, stc_n_msg_bits,stc_n_msg_bits_check,l_check_check);
        imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);    
      
%%  ��Ϣ��ȡ
        [stc_decoded_msg] = stc_extracting(afterchannel_stego_Path, stc_n_msg_bits,stc_n_msg_bits_check, C_QUANT,n,k,m,l_check_check);   
        
%%  ����ÿ��ͼ���������??      
        bit_error = double(raw_msg) - double(stc_decoded_msg);
        bit_error_number = sum(abs(bit_error));
        bit_error_rate(1,i_img) = bit_error_number/raw_msg_len;
        
%% ����ÿ�ų�����
%  ���ÿ��ͼ���������               
%         fprintf('%s\n',['payload: ',num2str(payload),'  image_number: ',index_str,'  error_rate: ',num2str(bit_error_rate(1,i_img))]);  %
    end
%  �������ͼ���ƽ��������
    ave_error_rate = mean(bit_error_rate);
    file_id = fopen('ddmr_log_ber.txt', 'a');
    
    fprintf(file_id,'image_size-%d-QF-%d-%s\n',image_size, QF, ['payload: ',num2str(payload),'  ave_error_rate: ',num2str(ave_error_rate)]);  
    fclose(file_id);
end
function save_msg(msg_path, msg, stc_msg, stc_c1, stc_c2)
save(msg_path,'msg', 'stc_msg', 'stc_c1', 'stc_c2');
end

function save_cost(cost_path, rho1_P, rho1_M)
save(cost_path, 'rho1_P', 'rho1_M');
end
function [rho1_P, rho1_M] = load_cost(cost_path)
cost_mat = load(cost_path);
rho1_P = cost_mat.rho1_P;
rho1_M = cost_mat.rho1_M;
end
