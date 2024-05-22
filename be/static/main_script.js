const start_button = document.getElementById('start');
start_button.addEventListener('click', function () {
    window.location.href = 'https://www.bilibili.com';
});

const button = document.getElementById('user');
const tooltip1 = document.getElementById('user_tooltip');
const tooltip2 = document.getElementById('plan_tooltip');
const tooltip3 = document.getElementById('help_tooltip');

function openTrainModelModal() {
    // 显示一个模态框或输入框以输入训练次数
    const trainingIterations = prompt("请输入训练次数：");

    if (trainingIterations !== null) {
        // 调用函数将训练次数上传到后端
        uploadTrainingIterations(trainingIterations);
    }
}

function uploadTrainingIterations(trainingIterations) {
    fetch('http://localhost:5000/trans/train_new', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'iterations': trainingIterations
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch((error) => {
        console.error('Error:', error);
    });
}

// button.addEventListener('mouseover',(event) => {

//     const { clientX, clientY } = event;
//     // 设置提示框的位置为鼠标的坐标
//     tooltip1.style.left = clientX + 'px';
//     tooltip1.style.top = clientY + 'px';
//     timer=setTimeout(()=>{
//         tooltip1.style.display = 'block'
//     }, 1000);
// });

// button.addEventListener('mouseout', () => {
//     clearTimeout(timer);
//     tooltip1.style.display = 'none';
// });

// const buttons = document.querySelectorAll('.btn');

// buttons.forEach(button => {
//     button.addEventListener('mouseover', (event) => {
//         const { clientX, clientY } = event;
//         const tooltip = button.querySelector('.tooltip');

//         // 设置提示框的位置为鼠标的坐标
//         tooltip.style.left = clientX + 'px';
//         tooltip.style.top = clientY + 'px';

//         timer = setTimeout(() => {
//             tooltip.style.display = 'block';
//         }, 1000);
//     });

//     button.addEventListener('mouseout', () => {
//         const tooltip = button.querySelector('.tooltip');
//         clearTimeout(timer);
//         tooltip.style.display = 'none';
//     });
// });
