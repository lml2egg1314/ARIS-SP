
warning off;
addpath('jpegtbx');
for i = 1:2
    
    for j = 1:7
        if i == 2 && j < 3
            continue;
        end

        QF = 55+10*i;
        attack_QF = 60+j*5;
        image_size = 256;
       
        tab_m = jpeg_qtable(QF);
        xm = 256;
        xn = 256;
        m_block = floor(xm/8);
        n_block = floor(xn/8);  
        rep_tab = repmat(tab_m, m_block, n_block);
        

        get_modification(QF, attack_QF, rep_tab);
 
    end
end

function  get_modification(QF, attack_QF, rep_tab)
    image_size = 256;
    base_dir = '/data/lml/watermarking';
    image_set = 'BB';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF);
    cost_dir = sprintf('%s-dmmr-cost', cover_dir); 
    recompress_dir=sprintf('%s/DMMR/recompress_modification_test', base_dir); 
    if ~exist(recompress_dir,'dir')
        mkdir(recompress_dir); 
    end
    
    cover_num = 10000; 
     
    robust_rate = zeros(cover_num,8,8);
    mod_prob_rate = zeros(cover_num,8,8);
%     payload = 1;
    
%%  ï¿½ï¿½Ï¢Ç¶ï¿½ï¿½    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        cost_path = fullfile([cost_dir,'/',num2str(i_img),'.mat']);
        recompress_path=fullfile([recompress_dir,'/',index_str,'.jpg']);
        imwrite(imread(cover_path),recompress_path,'quality',attack_QF);
        
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};
        nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));
       
        
        r_cover = dm(cover_path, rep_tab);
        r_recom = dm(recompress_path, rep_tab);
        diff = abs(r_cover - r_recom);
        diffs = (diff~=0);
        
        [rho1_P,rho1_M] = load_cost(cost_path);
        p = EmbeddingSimulator(C_COEFFS,rho1_P, rho1_M, nzAC);
        rr = zeros(8);
        mr = zeros(8);
        
        for r = 1:8
            for c = 1:8
                rr(r,c) = sum(sum(diffs(r:8:end,c:8:end)));
                mr(r,c) = sum(sum(p(r:8:end,c:8:end)));
            end
        end
        robust_rate(i_img,:,:) = rr;
        mod_prob_rate(i_img,:,:) = mr;
   
       
    end
    mean_robust_rate = mean(robust_rate,1);
    mean_mod_prob_rate = mean(mod_prob_rate,1);
    save_mat = sprintf('QF_%d_Attack_QF_%d.mat',QF, attack_QF);
    save(save_mat, 'mean_robust_rate','mean_mod_prob_rate');
end

function [rho1_P, rho1_M] = load_cost(cost_path)
cost_mat = load(cost_path);
rho1_P = cost_mat.rho1_P;
rho1_M = cost_mat.rho1_M;
end

function robust_elements = dm(image_path,rep_tab)
%% ÌáÈ¡ÔØÃÜÐòÁÐÔªËØ
    
    bits = 8;
    cover_spa = imread(image_path);
    cover_spa = double(cover_spa) - 2^(round(bits)-1);
%     [xm,xn] = size(cover_spa);
    t = dctmtx(8);
    fun = @(xl) (t*xl*(t'));
    cover_DCT = blkproc(cover_spa,[8 8],fun);%·Ö¿éDCT±ä»»
    
    robust_elements = round(cover_DCT./rep_tab);
    robust_elements = mod(robust_elements,2);
    

end
function p = EmbeddingSimulator(x, rho1_P, rho1_M, m)

    x = double(x);
    n = numel(x);
    
    lambda = calc_lambda(rho1_P, rho1_M, m, n);
    pChangeP1 = (exp(-lambda .* rho1_P))./(1 + exp(-lambda .* rho1_P) + exp(-lambda .* rho1_M));
    pChangeM1 = (exp(-lambda .* rho1_M))./(1 + exp(-lambda .* rho1_P) + exp(-lambda .* rho1_M));

    p = pChangeP1 + pChangeM1;
    function lambda = calc_lambda(rho1_P, rho1_M, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rho1_P))./(1 + exp(-l3 .* rho1_P) + exp(-l3 .* rho1_M));
            pM1 = (exp(-l3 .* rho1_M))./(1 + exp(-l3 .* rho1_P) + exp(-l3 .* rho1_M));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2; 
            pP1 = (exp(-lambda .* rho1_P))./(1 + exp(-lambda .* rho1_P) + exp(-lambda .* rho1_M));
            pM1 = (exp(-lambda .* rho1_M))./(1 + exp(-lambda .* rho1_P) + exp(-lambda .* rho1_M));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
        end
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        pP1 = pP1(:);
        pM1 = pM1(:);
        Ht = -(pP1.*log2(pP1))-(pM1.*log2(pM1))-((1-pP1-pM1).*log2(1-pP1-pM1));
        Ht(isnan(Ht)) = 0;
        Ht = sum(Ht);
    end

end