@tail -n +3 "%~f0" | jshell -
@goto :eof

import java.io.*;

try (
    var out = new FileOutputStream(new File("src/report/references.bib"));
) {
    for (var file: new File("src/report/bibliography").listFiles()) {
        new FileInputStream(file).transferTo(out);
        out.write('\n');
    }
}
    
