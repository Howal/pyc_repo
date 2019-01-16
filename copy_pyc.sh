python compile_all.py
find ./ -not -name '*.py' -not -name '.git*' -not -path "./output/*" -not -path "./external/*" -not -path "./data/*" -not -path "*git/*" -not -path "*idea/*" | xargs cp --parents -t ../pyc_repo
find ./ -path '*/symbols/*.py' | xargs cp --parents -t ../pyc_repo
