const pipeClassification = async (reads1, reads2, format) => {
  const inferResponse = await fetch('transcript', {
      method: 'POST',
      headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ "fformat": format, "reads_1": reads1, "reads_2": reads2 })
  }).then(response => {
       return response.blob().then((data) => {
          return {
            data: data,
            filename: response.headers.get('Content-disposition'),
          };
       });
    })
    .then(({ data, filename }) => {
        const downloadUrl = URL.createObjectURL(data);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = filename.split('=')[1];
        document.body.appendChild(a);
        a.click();
    })
};

const readsForm = document.getElementById('reads_form');

readsForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const reads1 = document.getElementById('input_seqs_r1');
    const reads2 = document.getElementById('input_seqs_r2');
    const format = document.getElementById('seqs_format');
    const response = document.getElementById('response_output');

    try {
        await pipeClassification(reads1.value, reads2.value, format.value);
    } catch (err) {
        console.error(err);
    }
});
