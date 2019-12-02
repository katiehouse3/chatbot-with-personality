//#4 Initialization
var botui = new BotUI('hello-world');
//#5 Start chat
botui.message.add({
    content: 'Hello World from bot! What is your name?'
}).then(function () {
    return botui.action.text({
        action: {
            placeholder: 'Your name'
        }
    });
}).then(function (res) {
    botui.message.add({
        content: 'Hello, ' + res.value + '!'
    });
});
