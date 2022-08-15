clc;
fprintf('compiling stc_ml_extract    ... ');
mex -output stc_ml_extract    mex_stc_ml_extract.cpp    stc_ml_c.cpp stc_embed_c.cpp stc_extract_c.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_pm1_pls_embed ... ');
mex -output stc_pm1_pls_embed mex_stc_pm1_pls_embed.cpp stc_ml_c.cpp stc_embed_c.cpp stc_extract_c.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_pm1_dls_embed ... ');
mex -output stc_pm1_dls_embed mex_stc_pm1_dls_embed.cpp stc_ml_c.cpp stc_embed_c.cpp stc_extract_c.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_pm2_pls_embed ... ');
mex -output stc_pm2_pls_embed mex_stc_pm2_pls_embed.cpp stc_ml_c.cpp stc_embed_c.cpp stc_extract_c.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_pm2_dls_embed ... ');
mex -output stc_pm2_dls_embed mex_stc_pm2_dls_embed.cpp stc_ml_c.cpp stc_embed_c.cpp stc_extract_c.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_embed         ... ');
mex -output stc_embed stc_embed.cpp common.cpp -I../include
fprintf('done\n');
fprintf('compiling stc_extract       ... ');
mex -output stc_extract stc_extract.cpp common.cpp -I../include
fprintf('done\n');

movefile('./*.mex*','../matlab');
fprintf('All compiled MEX files were moved to ../matlab folder.\n');