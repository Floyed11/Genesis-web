const start_button = document.getElementById('start');
start_button.addEventListener('click', function () {
    window.location.href = 'https://www.bilibili.com';
});

const button = document.getElementById('user');
const tooltip1 = document.getElementById('user_tooltip');
const tooltip2 = document.getElementById('plan_tooltip');
const tooltip3 = document.getElementById('help_tooltip');

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
