% viz_bna_verify_disagreement.m
%
% Render the BNA-bake-vs-authors'-annot disagreement map on fsaverage.
%
% Input MGHs (written by scripts/verify_bna_fsaverage_bake.py):
%   data/atlas/fsaverage_bake_fast2/verify_vs_published/
%     {lh,rh}.disagreement.smoothed.mgh   -- 0=agree/out-of-mask,
%                                            1=boundary disagreement,
%                                            2=interior disagreement.
%
% Shading recipe
%   Base cortex shaded by lh.curv (gyri light, sulci dark) so the cortical
%   topography is legible underneath the overlay. Then:
%     boundary disagreement (1) -> yellow
%     interior disagreement (2) -> red
%
% The interior-red points are the ones that matter: those are vertices
% >=1 vertex away from any published parcel boundary yet our argmax
% disagrees with the authors' label.
%
% Outputs under data/atlas/fsaverage_bake_fast2/verify_vs_published/figures/
%   {lh,rh}_disagreement_smoothed.png

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
in_dir    = fullfile(proj_root, 'data', 'atlas', 'fsaverage_bake_fast2', ...
                     'verify_vs_published');
out_dir   = fullfile(in_dir, 'figures');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'fsaverage', 'surf');

hemis = {'lh', 'rh'};
view_lateral = struct('lh', [270 0], 'rh', [90 0]);
view_medial  = struct('lh', [90 0],  'rh', [270 0]);
light_lateral = struct('lh', [-1 0 0], 'rh', [1 0 0]);
light_medial  = struct('lh', [1 0 0],  'rh', [-1 0 0]);

boundary_color = [1.00 0.85 0.10];   % yellow
interior_color = [0.95 0.15 0.15];   % red
base_sulc      = [0.55 0.55 0.55];
base_gyri      = [0.88 0.88 0.88];

for h = 1:numel(hemis)
    hemi = hemis{h};

    pial = fullfile(avg_surf_dir, sprintf('%s.pial', hemi));
    curv = fullfile(avg_surf_dir, sprintf('%s.curv', hemi));
    dis_mgh = fullfile(in_dir, sprintf('%s.disagreement.smoothed.mgh', hemi));

    fprintf('loading %s geometry + disagreement overlay\n', hemi);
    [verts, faces] = read_surf(pial);
    faces = faces + 1;
    nVert = size(verts, 1);

    cv = read_curv(curv);
    vcol = repmat(base_gyri, nVert, 1);
    vcol(cv > 0, :) = repmat(base_sulc, nnz(cv > 0), 1);

    dis_img = MRIread(dis_mgh);
    dis = squeeze(dis_img.vol);
    dis = dis(:);
    if numel(dis) ~= nVert
        error('%s disagreement has %d verts, expected %d', hemi, numel(dis), nVert);
    end

    n_bdy = nnz(dis == 1);
    n_int = nnz(dis == 2);
    fprintf('  boundary disagree: %d (%.1f%%)  interior disagree: %d (%.1f%%)\n', ...
            n_bdy, 100*n_bdy/nVert, n_int, 100*n_int/nVert);

    % Boundary: semi-blend with base (keeps curvature visible)
    bdy_mask = (dis == 1);
    vcol(bdy_mask, :) = 0.35 * vcol(bdy_mask, :) + 0.65 * repmat(boundary_color, nnz(bdy_mask), 1);
    % Interior: solid overlay (this is the load-bearing disagreement)
    int_mask = (dis == 2);
    vcol(int_mask, :) = repmat(interior_color, nnz(int_mask), 1);

    hFig = figure('Position', [100 100 1300 550], 'Color', 'w', ...
                  'Name', sprintf('BNA bake vs authors annot: %s', hemi));

    % lateral
    subplot(1, 2, 1);
    patch('Vertices', verts, 'Faces', faces, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull;
    axis off; axis equal; axis vis3d; hold on;
    light('Position', light_lateral.(hemi));
    view(view_lateral.(hemi));
    title(sprintf('%s lateral  (yellow=boundary %d, red=interior %d)', ...
                  hemi, n_bdy, n_int), 'Color', 'k', 'Interpreter', 'none');

    % medial
    subplot(1, 2, 2);
    patch('Vertices', verts, 'Faces', faces, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull;
    axis off; axis equal; axis vis3d; hold on;
    light('Position', light_medial.(hemi));
    view(view_medial.(hemi));
    title(sprintf('%s medial', hemi), 'Color', 'k', 'Interpreter', 'none');

    png = fullfile(out_dir, sprintf('%s_disagreement_smoothed.png', hemi));
    print(hFig, png, '-dpng', '-r150');
    fprintf('  wrote %s\n', png);
end

fprintf('\ndone. outputs in %s\n', out_dir);
