document.addEventListener('DOMContentLoaded', function () {
    const starsContainer = document.createElement('div');
    starsContainer.id = 'stars-container';
    document.body.prepend(starsContainer);

    const numberOfStars = 300;

    for (let i = 0; i < numberOfStars; i++) {
        let star = document.createElement('div');
        star.classList.add('star');
        let size = Math.random() * 2;
        star.style.width = `${size}px`;
        star.style.height = `${size}px`;
        star.style.left = `${Math.random() * 100}vw`;
        star.style.top = `${Math.random() * 200}vh`; // Start off-screen
        star.style.animationDuration = `${(Math.random() * 3) + 2}s`;
        star.style.animationDelay = `${Math.random() * 3}s`;
        starsContainer.appendChild(star);
    }
});