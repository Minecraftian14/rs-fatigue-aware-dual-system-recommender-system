@tail -n +3 "%~f0" | jshell -
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
System.out.println("""
\\begin{table}[!t]
\\caption{An Example of a Table\\label{tab:tableABC}}
\\centering
\\begin{tabular}{|l||c||c||c||c|}
\\hline
""".strip());

for (var line: text.split("[\\n\\r]+")) {
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