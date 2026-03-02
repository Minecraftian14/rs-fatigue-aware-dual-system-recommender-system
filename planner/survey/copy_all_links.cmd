@grep -Proh '(?^<=link: ).*' | @tail -n +2 | @clip
@echo Copied!