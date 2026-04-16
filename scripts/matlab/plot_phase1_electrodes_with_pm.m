% plot_phase1_electrodes_with_pm.m
%
% Like plot_phase1_electrodes.m, but also samples Brainnetome PM onto each
% lh.pial vertex and colors the cortex by PM total (hot colormap blended
% onto the gray base). Electrodes are scattered on top, unchanged.
%
% Inputs (expected on MATLAB path):
%   $FREESURFER_HOME/matlab          -> read_surf, MRIread
%   iELVis (for tripatchDG)          -> not strictly required here; we use
%                                       patch() directly to get per-vertex
%                                       color control.
%
% Data (expected on disk):
%   data/atlas/BNA_PM_total.nii.gz           (Brainnetome PM summed over 246 ROIs)
%   $SUBJECTS_DIR/cvs_avg35_inMNI152/surf/lh.pial
%   $SUBJECTS_DIR/cvs_avg35_inMNI152/mri/orig.mgz
%   data/mni_coords/<pt>_MNI152.csv          (electrode positions)

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
csv_dir   = fullfile(proj_root, 'data', 'mni_coords');
out_dir   = fullfile(csv_dir, 'viz_ielvis');
pm_path   = fullfile(proj_root, 'data', 'atlas', 'BNA_PM_total.nii.gz');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'cvs_avg35_inMNI152', 'surf');
avg_mri_dir  = fullfile(subjects_dir, 'cvs_avg35_inMNI152', 'mri');

% ---- load lh.pial ----
fprintf('loading lh.pial\n');
[lh_vert, lh_face] = read_surf(fullfile(avg_surf_dir, 'lh.pial'));
lh_face = lh_face + 1;  % read_surf returns 0-indexed faces
nVert = size(lh_vert, 1);
fprintf('  %d vertices, %d faces\n', nVert, size(lh_face, 1));

% ---- load PM volume ----
fprintf('loading Brainnetome PM total\n');
pm = MRIread(pm_path);
fprintf('  shape = %s   range = [%.1f, %.1f]\n', ...
    mat2str(size(pm.vol)), min(pm.vol(:)), max(pm.vol(:)));

% ---- figure out frame: tkrRAS vs scanner RAS ----
% Pial vertices for a FreeSurfer subject are canonically in tkrRAS.
% PM volume affine (pm.vox2ras0) maps voxels to scanner RAS.
% Convert vertices tkrRAS -> scanner RAS using the orig.mgz cras offset.
fprintf('loading orig.mgz to compute cras offset\n');
orig = MRIread(fullfile(avg_mri_dir, 'orig.mgz'));
% cras = scanner-RAS of the volume center (which equals tkrRAS origin)
%      = vox2ras0 * [ncols/2; nrows/2; nslices/2; 1]  (0-indexed centered voxel)
sz = size(orig.vol);
center_vox = [sz(2); sz(1); sz(3)] / 2;   % (c, r, s), 0-indexed
cras_h = orig.vox2ras0 * [center_vox; 1];
cras_offset = cras_h(1:3);
fprintf('  cras offset = [%.3f %.3f %.3f]\n', cras_offset(1), cras_offset(2), cras_offset(3));

% NOTE: If PM overlay looks offset after running once, flip this flag.
use_cras = true;
if use_cras
    vert_scanner = lh_vert + cras_offset';   % (nVert, 3)
else
    vert_scanner = lh_vert;
end

% ---- sample PM at each vertex (trilinear) ----
fprintf('sampling PM at %d vertices\n', nVert);
ras4 = [vert_scanner, ones(nVert, 1)]';                % 4 x nVert
vox  = pm.vox2ras0 \ ras4;                             % 4 x nVert, 0-indexed (c, r, s)
c = vox(1, :) + 1;                                     % -> 1-indexed
r = vox(2, :) + 1;
s = vox(3, :) + 1;
% interp3(V, Xq, Yq, Zq) queries V(Yq, Xq, Zq) when V is (row, col, slice).
% MRIread stores mri.vol as (row, col, slice), so X=c, Y=r, Z=s.
pm_vals = interp3(pm.vol, c, r, s, 'linear', 0);       % 1 x nVert
pm_vals = pm_vals(:);
fprintf('  PM on vertices: nonzero=%d/%d  max=%.1f  p95=%.1f\n', ...
    nnz(pm_vals > 0), nVert, max(pm_vals), prctile(pm_vals, 95));

% ---- build per-vertex color (gray base + hot overlay) ----
base_color  = [0.75 0.75 0.75];
vmax        = 200;          % PM_total p99 ~= 200, saturate there
thr         = 5;            % below this, stay fully gray
blend_max   = 0.85;         % max overlay opacity for PM >= vmax

pm_norm = max(0, min(1, (pm_vals - thr) / (vmax - thr)));   % [0,1]
hot_cmap = hot(256);
idx = round(pm_norm * 255) + 1;
idx(idx < 1)   = 1;
idx(idx > 256) = 256;
overlay = hot_cmap(idx, :);

alpha_blend = blend_max * (pm_vals > thr);
vcol = (1 - alpha_blend) .* repmat(base_color, nVert, 1) + ...
       alpha_blend        .* overlay;

% ---- patient config ----
patients = {'S14', 'S16', 'S23', 'S26', 'S33', 'S39', 'S62'};
colors = [ ...
    1.0 0.0 0.0;
    1.0 0.5 0.0;
    1.0 1.0 0.0;
    0.0 0.8 0.0;
    0.0 0.8 0.8;
    1.0 0.0 1.0;
    1.0 1.0 1.0];

% ---- per-patient plot on PM-colored pial ----
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_MNI152.csv', pt));
    if ~exist(csvFile, 'file')
        warning('missing %s -- skipping', csvFile);
        continue;
    end
    T = readtable(csvFile, 'TextType', 'string');
    xyz = [T.x, T.y, T.z];
    n   = size(xyz, 1);
    fprintf('plotting %s (%d electrodes)\n', pt, n);

    hFig = figure('Position', [100 100 1200 500], 'Color', 'w', 'Name', pt);

    subplot(1, 2, 1);
    patch('Vertices', lh_vert, 'Faces', lh_face, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
    light('Position', [-1 0 0]);
    view(270, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'none');
    title(sprintf('%s  lateral  (N=%d)', pt, n), 'Color', 'k');

    subplot(1, 2, 2);
    patch('Vertices', lh_vert, 'Faces', lh_face, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
    light('Position', [1 0 0]);
    view(90, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'none');
    title(sprintf('%s  medial', pt), 'Color', 'k');

    png = fullfile(out_dir, sprintf('%s_pm_lateral_medial.png', pt));
    print(hFig, png, '-dpng', '-r150');
    fprintf('  wrote %s\n', png);
end

% ---- combined all-patient lateral view on PM-colored pial ----
fprintf('plotting combined all-patient view\n');
hFig = figure('Position', [100 100 900 600], 'Color', 'w', 'Name', 'all patients + PM');
patch('Vertices', lh_vert, 'Faces', lh_face, ...
      'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
      'EdgeColor', 'none');
shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
light('Position', [-1 0 0]);
view(270, 0);

for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_MNI152.csv', pt));
    if ~exist(csvFile, 'file'); continue; end
    T = readtable(csvFile, 'TextType', 'string');
    scatter3(T.x, T.y, T.z, 16, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'none', ...
             'DisplayName', pt);
end
title('Phase-1 LH patients + Brainnetome PM on cvs\_avg35\_inMNI152 lh.pial', ...
      'Color', 'k');
png = fullfile(out_dir, 'all_patients_with_pm_lateral.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

fprintf('\ndone. outputs in %s\n', out_dir);
