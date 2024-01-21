import pathlib

def test_output_exist():
    flag_error = pathlib.Path("output.tsv")
    assert flag_error


def test_output_content():
    import filecmp
    filename_out = pathlib.Path("src/output.tsv")
    filename_ref = pathlib.Path("data/validation/subjects.tsv")
    
    print("Content of output file:", filename_out.read_text())
    print("Content of reference file:", filename_ref.read_text())

    # Read and compare the lines while ignoring leading/trailing whitespaces
    with open(filename_out, 'r') as f_out, open(filename_ref, 'r') as f_ref:
        lines_out = [line.strip() for line in f_out]
        lines_ref = [line.strip() for line in f_ref]

    assert lines_out==lines_ref, "Files are not identical"