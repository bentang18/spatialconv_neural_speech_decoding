% plot_phase1_electrodes.m
%
% Baseline sanity check for the v14 MNI152 electrode projection.
%
% Reads each Phase-1 LH patient's CSV from data/mni_coords/<pt>_MNI152.csv
% and renders the electrodes on cvs_avg35_inMNI152's lh.pial (real cortex
% with visible gyri and sulci) — matching Zac's published-figure style.
%
% IMPORTANT: the projection algorithm outputs coordinates on the
% outer-smoothed dural envelope (lh.pial-outer-smoothed), not on lh.pial.
% Electrodes will therefore hover ~2-4 mm above the real pial surface.
% That is correct: subdural ECoG grids physically sit on the dura, above
% the cortex, not inside the sulci. The visual gap == grid-to-cortex
% clearance, not a projection error.
%
% Expects these to already be set (see scripts/matlab/setup_ielvis.sh):
%   FREESURFER_HOME env var
%   SUBJECTS_DIR env var  (must contain cvs_avg35_inMNI152)
%   iELVis on the MATLAB path  (for tripatchDG)
%   $FREESURFER_HOME/matlab on the MATLAB path  (for read_surf)

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
csv_dir   = fullfile(proj_root, 'data', 'mni_coords');
out_dir   = fullfile(csv_dir, 'viz_ielvis');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'cvs_avg35_inMNI152', 'surf');

% ---- load lh.pial (real cortex) ONCE (same for every patient) ----
fprintf('loading cvs_avg35_inMNI152/surf/lh.pial\n');
[lh_vert, lh_face] = read_surf(fullfile(avg_surf_dir, 'lh.pial'));
lh_face = lh_face + 1;  % read_surf returns 0-indexed faces
avgPial.vert = lh_vert;
avgPial.tri  = lh_face;
fprintf('  %d vertices, %d faces\n', size(lh_vert, 1), size(lh_face, 1));

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

% ---- per-patient plot (lateral + medial, side by side) ----
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

    % --- lateral ---
    subplot(1, 2, 1);
    tripatchDG(avgPial, hFig, [0.75 0.75 0.75]);
    shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
    alpha(0.85);
    light('Position', [-1 0 0]);
    view(270, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'none');
    title(sprintf('%s  lateral  (N=%d)', pt, n), 'Color', 'k');

    % --- medial ---
    subplot(1, 2, 2);
    tripatchDG(avgPial, hFig, [0.75 0.75 0.75]);
    shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
    alpha(0.85);
    light('Position', [1 0 0]);
    view(90, 0);
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 18, ...
             'MarkerFaceColor', colors(k,:), ...
             'MarkerEdgeColor', 'none');
    title(sprintf('%s  medial', pt), 'Color', 'k');

    png = fullfile(out_dir, sprintf('%s_lateral_medial.png', pt));
    print(hFig, png, '-dpng', '-r150');
    fprintf('  wrote %s\n', png);
end

% ---- combined all-patient plot ----
fprintf('plotting combined all-patient lateral view\n');
hFig = figure('Position', [100 100 900 600], 'Color', 'w', 'Name', 'all patients');
tripatchDG(avgPial, hFig, [0.75 0.75 0.75]);
shading interp; lighting gouraud; material dull; axis off; axis equal; hold on;
alpha(0.85);
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
legend('Location', 'bestoutside', 'TextColor', 'k');
title('Phase-1 LH patients on cvs\_avg35\_inMNI152 (lh.pial-outer-smoothed)', ...
      'Color', 'k');

png = fullfile(out_dir, 'all_patients_lateral.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

fprintf('\ndone. outputs in %s\n', out_dir);
