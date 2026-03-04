@tail -n +3 "%~f0" | jshell - | clip
@goto :eof

import java.awt.*;
import java.awt.datatransfer.*;
var clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();

var data = clipboard.getContents(null);
if (data == null || !data.isDataFlavorSupported(DataFlavor.stringFlavor)) {
    System.err.println("Clipboard is empty or not a string.");
    System.exit(0);
}

var text = ((String) data.getTransferData(DataFlavor.stringFlavor)).trim();
var lines = text.split("[\\n\\r]+");

var expectedColumnCount = lines[0].split("\\|").length - 1;
var columnSetting = "|l|" + "|c|".repeat(expectedColumnCount - 1);

System.out.println("""
\\begin{table}[!t]
\\caption{An Example of a Table\\label{tab:tableABC}}
\\centering
\\begin{tabular}{%s}
\\hline
""".formatted(columnSetting).strip());

for (var line: lines) {
    var words = line.substring(1, line.length() - 1).split("\\|");
    for (int i = 0; i < words.length; i++) words[i] = words[i].strip();
    if (words[0].startsWith("-") && words[0].endsWith("-")) {
        System.out.println("\\hline"); 
        continue;
    }
    System.out.println(String.join(" & ", words) + " \\\\");
}

System.out.println("""
\\hline
\\end{tabular}
\\end{table}
""");