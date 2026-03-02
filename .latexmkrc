use Cwd;
my $root = getcwd();

# Output PDF
$pdf_mode = 1;

# # Use lualatex instead of pdflatex
# $pdflatex = 'lualatex -interaction=nonstopmode %O %S';
# $pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';

# Set source roots
@default_files = ('src/report/main.tex');

# Put build files in build/
$out_dir = "$root/build/report";
$aux_dir = "$root/build/report/temp";
$do_cd = 1;

# # Auto run biber (add \usepackage[backend=biber]{biblatex})
# $bibtex_use = 2;
# # $biber = 'biber %O %B';

# Continuous preview
$preview_continuous_mode = 1;
$pvc_view_file_via_temporary = 1;

# # Extend the list of files deleted when running `latexmk -c`
# $clean_ext .= ' %R.run.xml %R.bcf';