% plot_phase1_electrodes.m
%
% Phase-1 LH electrode positions on fsaverage lh.pial, post-#36.
%
% Pipeline recap (strict fsaverage snap-to-pial, #36 closed 2026-04-16):
%   patient lh.pial -> patient lh.sphere.reg -> fsaverage lh.sphere.reg
%   -> fsaverage lh.pial  (nearest vertex at each hop)
%
% Inputs
%   data/fsaverage_coords/<pt>_fsaverage_pial.csv
%       columns: name, hemisphere, fsaverage_vertex, x, y, z
%       (x, y, z) is already on fsaverage lh.pial; no snap at render time.
%   $SUBJECTS_DIR/fsaverage/surf/lh.pial   (geometry)
%   $SUBJECTS_DIR/fsaverage/surf/lh.curv   (binary sulcal/gyral shading)
%
% Outputs
%   data/fsaverage_coords/viz/<pt>_lateral_medial.png     per-patient
%   data/fsaverage_coords/viz/all_patients_lateral.png    pooled lateral
%   data/fsaverage_coords/viz/all_patients_medial.png     pooled medial
%
% Why the curvature shading: the whole point of verifying the fsaverage
% snap is to see whether electrodes end up on the gyral crowns they were
% placed on intra-operatively. A flat-gray cortex hides that. Binary
% curvature (gyri light, sulci dark) makes crown-vs-fundus placement
% visible at a glance.
%
% Expects (see scripts/matlab/setup_ielvis.sh):
%   FREESURFER_HOME env var
%   SUBJECTS_DIR env var  (must contain fsaverage)
%   $FREESURFER_HOME/matlab on the MATLAB path  (for read_surf, read_curv)
%   iELVis optional (we use patch() directly for per-vertex shading)

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
csv_dir   = fullfile(proj_root, 'data', 'fsaverage_coords');
out_dir   = fullfile(csv_dir, 'viz');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'fsaverage', 'surf');

% ---- load fsaverage lh.pial + curvature ----
fprintf('loading fsaverage/lh.pial + lh.curv\n');
[lh_vert, lh_face] = read_surf(fullfile(avg_surf_dir, 'lh.pial'));
lh_face = lh_face + 1;  % read_surf returns 0-indexed faces
nVert = size(lh_vert, 1);
fprintf('  %d vertices, %d faces\n', nVert, size(lh_face, 1));

curv = read_curv(fullfile(avg_surf_dir, 'lh.curv'));
% FreeSurfer mean-curv convention: curv > 0 = concave (sulcus),
% curv < 0 = convex (gyrus). Binary shade at 0 is the cleanest read
% for checking gyral-crown placement. Flip the sign below if your
% local lh.curv uses the inverted convention.
sulc_color = [0.55 0.55 0.55];   % dark gray = sulcus
gyr_color  = [0.88 0.88 0.88];   % light gray = gyrus
vcol = repmat(gyr_color, nVert, 1);
sulc_mask = curv > 0;
vcol(sulc_mask, :) = repmat(sulc_color, nnz(sulc_mask), 1);
fprintf('  curv range [%.3f, %.3f]   sulc vertices = %d (%.1f%%)\n', ...
        min(curv), max(curv), nnz(sulc_mask), 100 * nnz(sulc_mask) / nVert);

% ---- patient config ----
% Phase-1 LH patients with fsaverage coord caches (2026-04-16).
patients = {'S14', 'S16', 'S23', 'S26', 'S33', 'S39', 'S62'};
colors = [ ...
    0.95 0.10 0.10;   % S14  red
    1.00 0.55 0.00;   % S16  orange
    0.95 0.85 0.00;   % S23  yellow
    0.10 0.75 0.10;   % S26  green
    0.10 0.70 0.90;   % S33  cyan
    0.85 0.20 0.85;   % S39  magenta
    0.10 0.10 0.10];  % S62  black

% ---- per-patient lateral + medial ----
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_fsaverage_pial.csv', pt));
    if ~exist(csvFile, 'file')
        warning('missing %s -- skipping', csvFile);
        continue;
    end

    T = readtable(csvFile, 'TextType', 'string');
    T = T(strcmp(T.hemisphere, "L"), :);
    if isempty(T)
        warning('%s has no LH electrodes -- skipping', pt);
        continue;
    end
    xyz = [T.x, T.y, T.z];
    n   = size(xyz, 1);
    fprintf('plotting %s (%d LH electrodes)\n', pt, n);

    hFig = figure('Position', [100 100 1200 500], 'Color', 'w', 'Name', pt);

    subplot(1, 2, 1);
    patch('Vertices', lh_vert, 'Faces', lh_face, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull;
    axis off; axis equal; axis vis3d; hold on;
    light('Position', [-1 0 0]);
    view(270, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 20, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.3);
    title(sprintf('%s  lateral  (N=%d)', pt, n), 'Color', 'k');

    subplot(1, 2, 2);
    patch('Vertices', lh_vert, 'Faces', lh_face, ...
          'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
          'EdgeColor', 'none');
    shading interp; lighting gouraud; material dull;
    axis off; axis equal; axis vis3d; hold on;
    light('Position', [1 0 0]);
    view(90, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 20, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.3);
    title(sprintf('%s  medial', pt), 'Color', 'k');

    png = fullfile(out_dir, sprintf('%s_lateral_medial.png', pt));
    print(hFig, png, '-dpng', '-r150');
    fprintf('  wrote %s\n', png);
end

% ---- pooled all-patient lateral ----
fprintf('plotting pooled all-patient lateral view\n');
hFig = figure('Position', [100 100 1000 700], 'Color', 'w', ...
              'Name', 'all patients (fsaverage lateral)');
patch('Vertices', lh_vert, 'Faces', lh_face, ...
      'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
      'EdgeColor', 'none');
shading interp; lighting gouraud; material dull;
axis off; axis equal; axis vis3d; hold on;
light('Position', [-1 0 0]);
view(270, 0);

total_n = 0;
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_fsaverage_pial.csv', pt));
    if ~exist(csvFile, 'file'); continue; end
    T = readtable(csvFile, 'TextType', 'string');
    T = T(strcmp(T.hemisphere, "L"), :);
    if isempty(T); continue; end
    scatter3(T.x, T.y, T.z, 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.2, ...
             'DisplayName', pt);
    total_n = total_n + height(T);
end
legend('Location', 'bestoutside', 'TextColor', 'k');
title(sprintf('Phase-1 LH electrodes on fsaverage lh.pial (N=%d, pooled)', total_n), ...
      'Color', 'k', 'Interpreter', 'none');
png = fullfile(out_dir, 'all_patients_lateral.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

% ---- pooled all-patient medial ----
fprintf('plotting pooled all-patient medial view\n');
hFig = figure('Position', [100 100 1000 700], 'Color', 'w', ...
              'Name', 'all patients (fsaverage medial)');
patch('Vertices', lh_vert, 'Faces', lh_face, ...
      'FaceVertexCData', vcol, 'FaceColor', 'interp', ...
      'EdgeColor', 'none');
shading interp; lighting gouraud; material dull;
axis off; axis equal; axis vis3d; hold on;
light('Position', [1 0 0]);
view(90, 0);

for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_fsaverage_pial.csv', pt));
    if ~exist(csvFile, 'file'); continue; end
    T = readtable(csvFile, 'TextType', 'string');
    T = T(strcmp(T.hemisphere, "L"), :);
    if isempty(T); continue; end
    scatter3(T.x, T.y, T.z, 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.2, ...
             'DisplayName', pt);
end
legend('Location', 'bestoutside', 'TextColor', 'k');
title('Phase-1 LH electrodes on fsaverage lh.pial (pooled, medial)', ...
      'Color', 'k', 'Interpreter', 'none');
png = fullfile(out_dir, 'all_patients_medial.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

fprintf('\ndone. outputs in %s\n', out_dir);
